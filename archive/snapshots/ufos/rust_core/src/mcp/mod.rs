pub mod server;
pub mod resources;
pub mod tools;

use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize, Debug)]
pub struct McpRequest {
    pub method: String,
    pub params: serde_json::Value,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct McpResponse {
    pub result: serde_json::Value,
    pub error: Option<String>,
}
