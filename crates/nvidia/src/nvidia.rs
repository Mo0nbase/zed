mod supported_countries;

use anyhow::{anyhow, Result};
use futures::{io::BufReader, stream::BoxStream, AsyncBufReadExt, AsyncReadExt, Stream, StreamExt};
use http_client::{AsyncBody, HttpClient, HttpRequestExt, Method, Request as HttpRequest};
use serde::{Deserialize, Serialize};
use std::time::Duration;
use strum::EnumIter;

pub use supported_countries::*;

pub const NVIDIA_NIM_API_URL: &str = "https://integrate.api.nvidia.com/v1";

#[derive(Clone, Copy, Serialize, Deserialize, Debug, Eq, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    User,
    Assistant,
    System,
}

#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, EnumIter)]
pub enum Model {
    #[serde(
        rename = "nvidia/llama-3.1-nemotron-70b-instruct",
        alias = "nvidia/llama-3.1-nemotron-70b-instruct"
    )]
    #[default]
    Nemotron70BInstruct,

    #[serde(rename = "custom")]
    Custom {
        name: String,
        display_name: Option<String>,
        max_tokens: usize,
        max_output_tokens: Option<u32>,
    },
}

impl Model {
    pub fn from_id(id: &str) -> Result<Self> {
        match id {
            "nvidia/llama-3.1-nemotron-70b-instruct" => Ok(Self::Nemotron70BInstruct),
            _ => Err(anyhow!("invalid model id")),
        }
    }

    pub fn id(&self) -> &str {
        match self {
            Self::Nemotron70BInstruct => "nvidia/llama-3.1-nemotron-70b-instruct",
            Self::Custom { name, .. } => name,
        }
    }

    pub fn display_name(&self) -> &str {
        match self {
            Self::Nemotron70BInstruct => "Nemotron 70B Instruct",
            Self::Custom {
                name, display_name, ..
            } => display_name.as_ref().unwrap_or(name),
        }
    }

    pub fn max_token_count(&self) -> usize {
        match self {
            Self::Nemotron70BInstruct => 4096, // This value might need to be adjusted
            Self::Custom { max_tokens, .. } => *max_tokens,
        }
    }

    pub fn max_output_tokens(&self) -> Option<u32> {
        match self {
            Self::Custom {
                max_output_tokens, ..
            } => *max_output_tokens,
            _ => None,
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Request {
    pub model: String,
    pub messages: Vec<RequestMessage>,
    pub stream: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    pub temperature: f32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
}

#[derive(Serialize, Deserialize, Debug, Eq, PartialEq)]
pub struct RequestMessage {
    pub role: Role,
    pub content: String,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ResponseMessageDelta {
    pub role: Option<Role>,
    pub content: Option<String>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ChoiceDelta {
    pub index: u32,
    pub delta: ResponseMessageDelta,
    pub finish_reason: Option<String>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ResponseStreamEvent {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChoiceDelta>,
}

pub async fn stream_completion(
    client: &dyn HttpClient,
    api_url: &str,
    api_key: &str,
    request: Request,
    low_speed_timeout: Option<Duration>,
) -> Result<BoxStream<'static, Result<ResponseStreamEvent>>> {
    let uri = format!("{NVIDIA_NIM_API_URL}/chat/completions");
    let mut request_builder = HttpRequest::builder()
        .method(Method::POST)
        .uri(uri)
        .header("Content-Type", "application/json")
        .header("Authorization", format!("Bearer {}", api_key));

    if let Some(low_speed_timeout) = low_speed_timeout {
        request_builder = request_builder.read_timeout(low_speed_timeout);
    };

    println!("Sending request to NVIDIA API: {:?}", serde_json::to_string(&request));
    let request = request_builder.body(AsyncBody::from(serde_json::to_string(&request)?))?;

    println!("About to send request to NVIDIA API");

    let mut response = match client.send(request).await {
        Ok(resp) => {
            println!("Successfully sent request to NVIDIA API");
            resp
        },
        Err(e) => {
            eprintln!("Error sending request to NVIDIA API: {:?}", e);
            return Err(anyhow!("Failed to send request: {}", e));
        }
    };

    println!("Received response from NVIDIA API: {:?}", response.status());
    if response.status().is_success() {
        let reader = BufReader::new(response.into_body());
        Ok(reader
            .lines()
            .filter_map(|line| async move {
                match line {
                    Ok(line) => {
                        let line = line.strip_prefix("data: ")?;
                        if line == "[DONE]" {
                            None
                        } else {
                            match serde_json::from_str(line) {
                                Ok(response) => Some(Ok(response)),
                                Err(error) => Some(Err(anyhow!(error))),
                            }
                        }
                    }
                    Err(error) => Some(Err(anyhow!(error))),
                }
            })
            .boxed())
    } else {
        let mut body = String::new();
        response.body_mut().read_to_string(&mut body).await?;
        Err(anyhow!(
            "Failed to connect to NVIDIA NIM API: {} {}",
            response.status(),
            body,
        ))
    }
}

pub fn extract_text_from_events(
    response: impl Stream<Item = Result<ResponseStreamEvent>>,
) -> impl Stream<Item = Result<String>> {
    response.filter_map(|response| async move {
        match response {
            Ok(mut response) => Some(Ok(response.choices.pop()?.delta.content?)),
            Err(error) => Some(Err(error)),
        }
    })
}
