
use axum::{
    routing::{get, post},
    Router,
    Json,
};
use std::net::SocketAddr;
use crate::mcp::{McpRequest, McpResponse};

// 4.1 MCP Server Architecture in Rust
// Supports SSE (Remote) via Axum

pub async fn start_mcp_server(port: u16) {
    let app = Router::new()
        .route("/mcp", post(handle_mcp_request))
        .route("/sse", get(sse_handler));

    let addr = SocketAddr::from(([127, 0, 0, 1], port));
    println!("MCP Server listening on {}", addr);

    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

async fn handle_mcp_request(Json(payload): Json<McpRequest>) -> Json<McpResponse> {
    // Dispatch to tools or resources
    Json(McpResponse {
        result: serde_json::json!({"status": "ok"}),
        error: None,
    })
}

async fn sse_handler() -> &'static str {
    "Stream..."
}
