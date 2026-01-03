use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use tokio::sync::mpsc;

// Unified Ledger Order Schema Hierarchy (Table 1)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IntentSide {
    Buy,
    Sell,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Order {
    pub order_id: String,       // UUID
    pub parent_id: Option<String>, // Reference to aggregate strategy order
    pub client_id: String,      // WM: Client Portfolio
    pub desk_id: String,        // IB: Trading Desk
    pub strategy_tag: Option<String>, // AM/IB: Algo Strategy ID
    pub intent_side: IntentSide,
    pub symbol: String,
    pub price: f64,
    pub quantity: f64,
    pub internalization_flag: bool,
}

struct OrderBook {
    bids: Vec<Order>,
    asks: Vec<Order>,
}

impl OrderBook {
    fn new() -> Self {
        OrderBook {
            bids: Vec::new(),
            asks: Vec::new(),
        }
    }

    fn add_order(&mut self, mut order: Order) {
        println!("Iron Core: Ingesting Order {}", order.order_id);

        // Internalization Logic (Simplified)
        if self.try_internalize(&mut order) {
            println!("Iron Core: Order {} Internalized", order.order_id);
            order.internalization_flag = true;
            // In a real system, we would persist the fill to the ledger here
        } else {
            match order.intent_side {
                IntentSide::Buy => self.bids.push(order),
                IntentSide::Sell => self.asks.push(order),
            }
        }
    }

    fn try_internalize(&mut self, incoming_order: &mut Order) -> bool {
        // Mock logic: Check if we have a crossing order in the book
        // Real logic would check NBBO and cross at midpoint
        false
    }
}

struct IronCore {
    books: HashMap<String, Arc<Mutex<OrderBook>>>,
}

impl IronCore {
    fn new() -> Self {
        IronCore {
            books: HashMap::new(),
        }
    }

    fn get_book(&mut self, symbol: &str) -> Arc<Mutex<OrderBook>> {
        self.books
            .entry(symbol.to_string())
            .or_insert_with(|| Arc::new(Mutex::new(OrderBook::new())))
            .clone()
    }
}

#[tokio::main]
async fn main() {
    println!("Starting Adam v24.0 Iron Core...");

    let core = Arc::new(Mutex::new(IronCore::new()));

    // Simulation of MCP Resource Stream or Redis Subscription
    println!("Listening for high-frequency market data...");

    // Mock event loop
    let symbols = vec!["AAPL", "GOOGL", "MSFT"];

    for symbol in symbols {
        let core_ref = core.clone();
        let sym = symbol.to_string();
        tokio::spawn(async move {
            let mut locked_core = core_ref.lock().unwrap();
            let _book = locked_core.get_book(&sym);
            println!("Initialized OrderBook for {}", sym);
        });
    }

    // Keep alive
    loop {
        tokio::time::sleep(tokio::time::Duration::from_secs(5)).await;
        println!("Iron Core Heartbeat: System Nominal");
    }
}
