use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemoryProvenance {
    pub activity_id: String,
    pub agent_id: String,
    pub policy_version: String,
    #[serde(default)]
    pub parent_record_ids: Vec<String>,
}

impl MemoryProvenance {
    pub fn is_valid(&self) -> bool {
        !self.activity_id.trim().is_empty()
            && !self.agent_id.trim().is_empty()
            && !self.policy_version.trim().is_empty()
    }
}
