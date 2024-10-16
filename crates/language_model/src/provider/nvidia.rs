use crate::LanguageModelCompletionEvent;
use crate::StopReason;
use crate::{
    settings::AllLanguageModelSettings, LanguageModel, LanguageModelCacheConfiguration,
    LanguageModelId, LanguageModelName, LanguageModelProvider, LanguageModelProviderId,
    LanguageModelProviderName, LanguageModelProviderState, LanguageModelRequest, RateLimiter, Role,
};
use anyhow::{anyhow, Result};
use editor::{Editor, EditorElement, EditorStyle};
use futures::Stream;
use futures::{future::BoxFuture, stream::BoxStream, FutureExt, StreamExt};
use gpui::{
    AnyView, AppContext, AsyncAppContext, FontStyle, ModelContext, Subscription, Task, TextStyle,
    View, WhiteSpace,
};
use http_client::HttpClient;
use nvidia::{stream_completion, Model, Request, RequestMessage, ResponseStreamEvent};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use settings::{Settings, SettingsStore};
use std::{fmt, sync::Arc, time::Duration};
use theme::ThemeSettings;
use ui::{prelude::*, Icon, IconName, Tooltip};
use util::ResultExt;

const PROVIDER_ID: &str = "nvidia";
const PROVIDER_NAME: &str = "NVIDIA";

#[derive(Default, Clone, Debug, PartialEq)]
pub struct NvidiaSettings {
    pub api_url: String,
    pub low_speed_timeout: Option<Duration>,
    pub available_models: Vec<AvailableModel>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct AvailableModel {
    pub name: String,
    pub display_name: Option<String>,
    pub max_tokens: usize,
    pub max_output_tokens: Option<u32>,
}

pub struct NvidiaLanguageModelProvider {
    http_client: Arc<dyn HttpClient>,
    state: gpui::Model<State>,
}

const NVIDIA_API_KEY_VAR: &str = "NVIDIA_API_KEY";

pub struct State {
    api_key: Option<String>,
    api_key_from_env: bool,
    _subscription: Subscription,
}

impl State {
    fn is_authenticated(&self) -> bool {
        self.api_key.is_some()
    }

    fn reset_api_key(&self, cx: &mut ModelContext<Self>) -> Task<Result<()>> {
        let settings = &AllLanguageModelSettings::get_global(cx).nvidia;
        let delete_credentials = cx.delete_credentials(&settings.api_url);
        cx.spawn(|this, mut cx| async move {
            delete_credentials.await.log_err();
            this.update(&mut cx, |this, cx| {
                this.api_key = None;
                this.api_key_from_env = false;
                cx.notify();
            })
        })
    }

    fn set_api_key(&mut self, api_key: String, cx: &mut ModelContext<Self>) -> Task<Result<()>> {
        let settings = &AllLanguageModelSettings::get_global(cx).nvidia;
        let write_credentials =
            cx.write_credentials(&settings.api_url, "Bearer", api_key.as_bytes());

        cx.spawn(|this, mut cx| async move {
            write_credentials.await?;
            this.update(&mut cx, |this, cx| {
                this.api_key = Some(api_key);
                cx.notify();
            })
        })
    }

    fn authenticate(&self, cx: &mut ModelContext<Self>) -> Task<Result<()>> {
        if self.is_authenticated() {
            Task::ready(Ok(()))
        } else {
            let api_url = AllLanguageModelSettings::get_global(cx)
                .nvidia
                .api_url
                .clone();
            cx.spawn(|this, mut cx| async move {
                let (api_key, from_env) = if let Ok(api_key) = std::env::var(NVIDIA_API_KEY_VAR) {
                    (api_key, true)
                } else {
                    let (_, api_key) = cx
                        .update(|cx| cx.read_credentials(&api_url))?
                        .await?
                        .ok_or_else(|| anyhow!("credentials not found"))?;
                    (String::from_utf8(api_key)?, false)
                };
                this.update(&mut cx, |this, cx| {
                    this.api_key = Some(api_key);
                    this.api_key_from_env = from_env;
                    cx.notify();
                })
            })
        }
    }
}

impl NvidiaLanguageModelProvider {
    pub fn new(http_client: Arc<dyn HttpClient>, cx: &mut AppContext) -> Self {
        let state = cx.new_model(|cx| State {
            api_key: None,
            api_key_from_env: false,
            _subscription: cx.observe_global::<SettingsStore>(|_, cx| {
                cx.notify();
            }),
        });

        Self { http_client, state }
    }
}

impl LanguageModelProviderState for NvidiaLanguageModelProvider {
    type ObservableEntity = State;

    fn observable_entity(&self) -> Option<gpui::Model<Self::ObservableEntity>> {
        Some(self.state.clone())
    }
}

impl LanguageModelProvider for NvidiaLanguageModelProvider {
    fn id(&self) -> LanguageModelProviderId {
        LanguageModelProviderId(PROVIDER_ID.into())
    }

    fn name(&self) -> LanguageModelProviderName {
        LanguageModelProviderName(PROVIDER_NAME.into())
    }

    fn icon(&self) -> IconName {
        IconName::AiNvidia // Ensure this icon exists
    }

    fn provided_models(&self, cx: &AppContext) -> Vec<Arc<dyn LanguageModel>> {
        let mut models = vec![Model::Nemotron70BInstruct];

        // Add custom models from settings
        for model in AllLanguageModelSettings::get_global(cx)
            .nvidia
            .available_models
            .iter()
        {
            models.push(Model::Custom {
                name: model.name.clone(),
                display_name: model.display_name.clone(),
                max_tokens: model.max_tokens,
                max_output_tokens: model.max_output_tokens,
            });
        }

        models
            .into_iter()
            .map(|model| {
                Arc::new(NvidiaModel {
                    id: LanguageModelId::from(model.id().to_string()),
                    model,
                    state: self.state.clone(),
                    http_client: self.http_client.clone(),
                    request_limiter: RateLimiter::new(4),
                }) as Arc<dyn LanguageModel>
            })
            .collect()
    }

    fn is_authenticated(&self, cx: &AppContext) -> bool {
        self.state.read(cx).is_authenticated()
    }

    fn authenticate(&self, cx: &mut AppContext) -> Task<Result<()>> {
        self.state.update(cx, |state, cx| state.authenticate(cx))
    }

    fn configuration_view(&self, cx: &mut WindowContext) -> AnyView {
        cx.new_view(|cx| ConfigurationView::new(self.state.clone(), cx))
            .into()
    }

    fn reset_credentials(&self, cx: &mut AppContext) -> Task<Result<()>> {
        self.state.update(cx, |state, cx| state.reset_api_key(cx))
    }
}

pub struct NvidiaModel {
    id: LanguageModelId,
    model: Model,
    state: gpui::Model<State>,
    http_client: Arc<dyn HttpClient>,
    request_limiter: RateLimiter,
}

impl NvidiaModel {
    fn get_request_info(&self, cx: &AppContext) -> (String, String, Option<Duration>) {
        let settings = AllLanguageModelSettings::get_global(cx).nvidia.clone();
        let api_url = settings.api_url;
        let low_speed_timeout = settings.low_speed_timeout;
        let api_key = self.state.read(cx).api_key.clone().unwrap_or_default();
        (api_url, api_key, low_speed_timeout)
    }

    fn stream_completion(
        &self,
        request: nvidia::Request,
        cx: &AsyncAppContext,
    ) -> BoxFuture<'static, Result<BoxStream<'static, Result<ResponseStreamEvent>>>> {
        let http_client = self.http_client.clone();
        let state = self.state.clone();

        let Ok((api_key, api_url, low_speed_timeout)) = cx.read_model(&state, |state, cx| {
            let settings = &AllLanguageModelSettings::get_global(cx).nvidia;
            (
                state.api_key.clone(),
                settings.api_url.clone(),
                settings.low_speed_timeout,
            )
        }) else {
            return futures::future::ready(Err(anyhow!("App state dropped"))).boxed();
        };

        let future = self.request_limiter.stream(async move {
            let api_key = api_key.ok_or_else(|| anyhow!("Missing NVIDIA API Key"))?;
            let response = stream_completion(
                http_client.as_ref(),
                &api_url,
                &api_key,
                request,
                low_speed_timeout,
            )
            .await?;

            // Add debug logging for the response
            println!("Received response from NVIDIA API");

            // Wrap the response stream to log each event
            let logged_response = response.inspect(|event| match event {
                Ok(event) => println!("Received event from NVIDIA API: {:?}", event),
                Err(e) => eprintln!("Error in NVIDIA API response stream: {:?}", e),
            });

            Ok(logged_response.boxed())
        });

        async move { Ok(future.await?.boxed()) }.boxed()
    }
}

impl LanguageModel for NvidiaModel {
    fn id(&self) -> LanguageModelId {
        self.id.clone()
    }

    fn name(&self) -> LanguageModelName {
        LanguageModelName::from(self.model.display_name().to_string())
    }

    fn provider_id(&self) -> LanguageModelProviderId {
        LanguageModelProviderId(PROVIDER_ID.into())
    }

    fn provider_name(&self) -> LanguageModelProviderName {
        LanguageModelProviderName(PROVIDER_NAME.into())
    }

    fn telemetry_id(&self) -> String {
        format!("nvidia/{}", self.model.id())
    }

    fn max_token_count(&self) -> usize {
        self.model.max_token_count()
    }

    fn max_output_tokens(&self) -> Option<u32> {
        self.model.max_output_tokens()
    }

    fn count_tokens(
        &self,
        request: LanguageModelRequest,
        cx: &AppContext,
    ) -> BoxFuture<'static, Result<usize>> {
        count_nvidia_tokens(request, self.model.clone(), cx)
    }

    fn stream_completion(
        &self,
        request: LanguageModelRequest,
        cx: &AsyncAppContext,
    ) -> BoxFuture<
        'static,
        Result<futures::stream::BoxStream<'static, Result<LanguageModelCompletionEvent>>>,
    > {
        let request = request.into_nvidia(self.model.id().into(), Some(1024));
        let completions = self.stream_completion(request, cx);
        async move { Ok(map_to_language_model_completion_events(completions.await?).boxed()) }
            .boxed()
    }

    fn cache_configuration(&self) -> Option<LanguageModelCacheConfiguration> {
        None // Implement if NVIDIA supports caching
    }

    fn use_any_tool(
        &self,
        _request: LanguageModelRequest,
        _tool_name: String,
        _tool_description: String,
        _schema: serde_json::Value,
        _cx: &AsyncAppContext,
    ) -> BoxFuture<'static, Result<futures::stream::BoxStream<'static, Result<String>>>> {
        // Implement NVIDIA-specific tool use logic here
        // This is a placeholder implementation
        async move { Err(anyhow!("Tool use not implemented for NVIDIA models")) }.boxed()
    }
}

pub fn map_to_language_model_completion_events(
    events: BoxStream<'static, Result<ResponseStreamEvent>>,
) -> impl Stream<Item = Result<LanguageModelCompletionEvent>> {
    events.filter_map(|event| async move {
        match event {
            Ok(event) => {
                println!("Processing event: {:?}", event); // Debug logging
                let choice = match event.choices.into_iter().next() {
                    Some(choice) => choice,
                    None => {
                        println!("No choices in response");
                        return None;
                    }
                };
                
                match (choice.delta.content, choice.finish_reason) {
                    (Some(content), _) => Some(Ok(LanguageModelCompletionEvent::Text(content))),
                    (None, Some(finish_reason)) => {
                        let stop_reason = match finish_reason.as_str() {
                            "stop" => StopReason::EndTurn,
                            "length" => StopReason::MaxTokens,
                            _ => StopReason::EndTurn,
                        };
                        Some(Ok(LanguageModelCompletionEvent::Stop(stop_reason)))
                    },
                    (None, None) => {
                        println!("Received role-only event"); // Debug logging
                        None // Ignore this event
                    }
                }
            },
            Err(e) => {
                eprintln!("Error processing event: {:?}", e);
                Some(Err(e))
            }
        }
    })
}

struct ConfigurationView {
    api_key_editor: View<Editor>,
    state: gpui::Model<State>,
    load_credentials_task: Option<Task<()>>,
}

impl ConfigurationView {
    fn new(state: gpui::Model<State>, cx: &mut ViewContext<Self>) -> Self {
        let api_key_editor = cx.new_view(|cx| {
            let mut editor = Editor::single_line(cx);
            editor.set_placeholder_text("Enter your NVIDIA API key here", cx);
            editor
        });

        cx.observe(&state, |_, _, cx| {
            cx.notify();
        })
        .detach();

        let load_credentials_task = Some(cx.spawn({
            let state = state.clone();
            |this, mut cx| async move {
                if let Some(task) = state
                    .update(&mut cx, |state, cx| state.authenticate(cx))
                    .log_err()
                {
                    let _ = task.await;
                }

                this.update(&mut cx, |this, cx| {
                    this.load_credentials_task = None;
                    cx.notify();
                })
                .log_err();
            }
        }));

        Self {
            api_key_editor,
            state,
            load_credentials_task,
        }
    }

    fn save_api_key(&mut self, _: &menu::Confirm, cx: &mut ViewContext<Self>) {
        let api_key = self.api_key_editor.read(cx).text(cx);
        if api_key.is_empty() {
            return;
        }

        let state = self.state.clone();
        cx.spawn(|_, mut cx| async move {
            state
                .update(&mut cx, |state, cx| state.set_api_key(api_key, cx))?
                .await
        })
        .detach_and_log_err(cx);

        cx.notify();
    }

    fn reset_api_key(&mut self, cx: &mut ViewContext<Self>) {
        self.api_key_editor
            .update(cx, |editor, cx| editor.set_text("", cx));

        let state = self.state.clone();
        cx.spawn(|_, mut cx| async move {
            state
                .update(&mut cx, |state, cx| state.reset_api_key(cx))?
                .await
        })
        .detach_and_log_err(cx);

        cx.notify();
    }

    fn render_api_key_editor(&self, cx: &mut ViewContext<Self>) -> impl IntoElement {
        let settings = ThemeSettings::get_global(cx);
        let text_style = TextStyle {
            color: cx.theme().colors().text,
            font_family: settings.ui_font.family.clone(),
            font_features: settings.ui_font.features.clone(),
            font_fallbacks: settings.ui_font.fallbacks.clone(),
            font_size: rems(0.875).into(),
            font_weight: settings.ui_font.weight,
            font_style: FontStyle::Normal,
            line_height: relative(1.3),
            background_color: None,
            underline: None,
            strikethrough: None,
            white_space: WhiteSpace::Normal,
            truncate: None,
        };
        EditorElement::new(
            &self.api_key_editor,
            EditorStyle {
                background: cx.theme().colors().editor_background,
                local_player: cx.theme().players().local(),
                text: text_style,
                ..Default::default()
            },
        )
    }

    fn should_render_editor(&self, cx: &mut ViewContext<Self>) -> bool {
        !self.state.read(cx).is_authenticated()
    }
}

impl Render for ConfigurationView {
    fn render(&mut self, cx: &mut ViewContext<Self>) -> impl IntoElement {
        const NVIDIA_CONSOLE_URL: &str = "https://www.nvidia.com/en-us/gpu-cloud/";
        const INSTRUCTIONS: [&str; 4] = [
            "To use Zed's assistant with NVIDIA, you need to add an API key. Follow these steps:",
            " - Create one by visiting:",
            " - Ensure your NVIDIA account has the necessary permissions",
            " - Paste your API key below and hit enter to start using the assistant",
        ];

        let env_var_set = self.state.read(cx).api_key_from_env;

        if self.load_credentials_task.is_some() {
            div().child(Label::new("Loading credentials...")).into_any()
        } else if self.should_render_editor(cx) {
            v_flex()
                .size_full()
                .on_action(cx.listener(Self::save_api_key))
                .child(Label::new(INSTRUCTIONS[0]))
                .child(
                    h_flex()
                        .child(Label::new(INSTRUCTIONS[1]))
                        .child(
                            Button::new("nvidia_console", NVIDIA_CONSOLE_URL)
                                .style(ButtonStyle::Subtle)
                                .icon(IconName::ExternalLink)
                                .icon_size(IconSize::XSmall)
                                .icon_color(Color::Muted)
                                .on_click(move |_, cx| cx.open_url(NVIDIA_CONSOLE_URL)),
                        ),
                )
                .children(
                    (2..INSTRUCTIONS.len()).map(|n| Label::new(INSTRUCTIONS[n])).collect::<Vec<_>>(),
                )
                .child(
                    h_flex()
                        .w_full()
                        .my_2()
                        .px_2()
                        .py_1()
                        .bg(cx.theme().colors().editor_background)
                        .rounded_md()
                        .child(self.render_api_key_editor(cx)),
                )
                .child(
                    Label::new(
                        format!(
                            "You can also assign the {NVIDIA_API_KEY_VAR} environment variable and restart Zed."
                        ),
                    )
                    .size(LabelSize::Small),
                )
                .into_any()
        } else {
            h_flex()
                .size_full()
                .justify_between()
                .child(
                    h_flex()
                        .gap_1()
                        .child(Icon::new(IconName::Check).color(Color::Success))
                        .child(Label::new(if env_var_set {
                            format!("API key set in {NVIDIA_API_KEY_VAR} environment variable.")
                        } else {
                            "API key configured.".to_string()
                        })),
                )
                .child(
                    Button::new("reset-key", "Reset key")
                        .icon(Some(IconName::Trash))
                        .icon_size(IconSize::Small)
                        .icon_position(IconPosition::Start)
                        .disabled(env_var_set)
                        .when(env_var_set, |this| {
                            this.tooltip(|cx| {
                                Tooltip::text(
                                    format!(
                                        "To reset your API key, unset the {NVIDIA_API_KEY_VAR} environment variable."
                                    ),
                                    cx,
                                )
                            })
                        })
                        .on_click(cx.listener(|this, _, cx| this.reset_api_key(cx))),
                )
                .into_any()
        }
    }
}

pub fn count_nvidia_tokens(
    request: LanguageModelRequest,
    _model: Model,
    cx: &AppContext,
) -> BoxFuture<'static, Result<usize>> {
    cx.background_executor()
        .spawn(async move {
            // Implement NVIDIA-specific token counting logic here
            // This is a placeholder implementation
            let total_chars = request
                .messages
                .iter()
                .map(|m| m.content.len())
                .sum::<usize>(); // Assuming m.content is String
            Ok(total_chars / 4) // Assuming 1 token is roughly 4 characters
        })
        .boxed()
}
