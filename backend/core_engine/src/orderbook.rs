use std::collections::BTreeMap;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Order {
    pub id: String,
    pub symbol: String,
    pub side: String, // "BUY" or "SELL"
    pub price: f64,
    pub quantity: f64,
    pub timestamp: i64,
}

#[derive(Debug, Default)]
pub struct OrderBook {
    pub symbol: String,
    pub bids: BTreeMap<String, Vec<Order>>, // Price -> Orders (Reverse for bids usually handled by iterator)
    pub asks: BTreeMap<String, Vec<Order>>, // Price -> Orders
}

impl OrderBook {
    pub fn new(symbol: &str) -> Self {
        OrderBook {
            symbol: symbol.to_string(),
            bids: BTreeMap::new(),
            asks: BTreeMap::new(),
        }
    }

    pub fn add_order(&mut self, order: Order) {
        // Simple implementation - just storing
        let price_key = format!("{:.2}", order.price); // String key for now to simplify
        if order.side == "BUY" {
            self.bids.entry(price_key).or_insert_with(Vec::new).push(order);
        } else {
            self.asks.entry(price_key).or_insert_with(Vec::new).push(order);
        }
    }
}
