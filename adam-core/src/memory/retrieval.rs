use crate::memory::admission::{AdmissionEngine, MemoryNode, MemoryTier};

/// Deterministic memory retriever that preserves original epistemic status.
pub struct MemoryRetriever<'a> {
    engine: &'a AdmissionEngine,
}

impl<'a> MemoryRetriever<'a> {
    pub fn new(engine: &'a AdmissionEngine) -> Self {
        MemoryRetriever { engine }
    }

    pub fn get_by_id(&self, memory_id: &str) -> Option<&'a MemoryNode> {
        self.engine.get_node(memory_id)
    }

    pub fn filter_by_tier(&self, tier: MemoryTier) -> Vec<&'a MemoryNode> {
        self.engine
            .list_nodes()
            .into_iter()
            .filter(|n| n.tier == tier)
            .collect()
    }

    pub fn filter_by_correlation_id(&self, correlation_id: &str) -> Vec<&'a MemoryNode> {
        self.engine
            .list_nodes()
            .into_iter()
            .filter(|n| n.correlation_id == correlation_id)
            .collect()
    }
}
