mod ledger;
mod matching_engine;
mod risk;

fn main() {
    println!("Starting UFOS v30 Core Engine...");
    println!("Initializing Unified Ledger...");
    let _ledger = ledger::UnifiedLedger::new();
    println!("Starting Matching Engine...");
    // In a real implementation, this would start the Tokio runtime and listen for MCP messages
}
