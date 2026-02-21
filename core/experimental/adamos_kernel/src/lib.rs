use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;
use std::sync::{Arc, Mutex};

/// Represents an Agent in the WASM sandbox
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Agent {
    pub id: String,
    pub name: String,
    pub capabilities: Vec<String>,
    pub status: AgentStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AgentStatus {
    Idle,
    Thinking,
    Sleeping,
    Dead,
}

/// A message passed between agents via the Kernel
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub id: String,
    pub sender: String,
    pub recipient: String,
    pub content: String,
    pub timestamp: u64,
}

/// The AdamOS Kernel (Orchestrator)
pub struct Kernel {
    pub agents: Arc<Mutex<HashMap<String, Agent>>>,
    pub message_bus: Arc<Mutex<Vec<Message>>>,
}

impl Kernel {
    pub fn new() -> Self {
        Kernel {
            agents: Arc::new(Mutex::new(HashMap::new())),
            message_bus: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Registers a new agent (e.g., loaded from WASM)
    pub fn register_agent(&self, name: String, capabilities: Vec<String>) -> String {
        let id = Uuid::new_v4().to_string();
        let agent = Agent {
            id: id.clone(),
            name,
            capabilities,
            status: AgentStatus::Idle,
        };

        let mut agents = self.agents.lock().unwrap();
        agents.insert(id.clone(), agent);
        println!("Kernel: Registered Agent {} ({})", name, id);
        id
    }

    /// Routes a message
    pub fn send_message(&self, sender_id: &str, recipient_id: &str, content: String) {
        let msg = Message {
            id: Uuid::new_v4().to_string(),
            sender: sender_id.to_string(),
            recipient: recipient_id.to_string(),
            content,
            timestamp: 0, // Mock timestamp
        };

        let mut bus = self.message_bus.lock().unwrap();
        bus.push(msg);
        println!("Kernel: Routed message from {} to {}", sender_id, recipient_id);
    }

    /// Simulates a kernel tick (routing, health checks)
    pub fn tick(&self) {
        let agents = self.agents.lock().unwrap();
        println!("Kernel: Health Check - {} Agents Alive", agents.len());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kernel_boot() {
        let kernel = Kernel::new();
        let id = kernel.register_agent("RiskAgent".to_string(), vec!["calculate_var".to_string()]);
        assert!(!id.is_empty());
    }
}
