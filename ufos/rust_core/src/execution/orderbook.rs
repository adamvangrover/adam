use super::{Order, Side};
use std::collections::BTreeMap;

// 2.2 The Lock-Free Order Book
// Note: Implementing a truly lock-free orderbook in safe Rust for a blueprint is complex.
// We will use a RwLock for simplicity in the prototype, but structure it for the LMAX pattern.

pub struct OrderBook {
    pub bids: BTreeMap<u64, Vec<Order>>, // Price -> Orders (descending)
    pub asks: BTreeMap<u64, Vec<Order>>, // Price -> Orders (ascending)
    // In a real lock-free implementation, these would be managed via atomic pointers or the ring buffer
}

impl OrderBook {
    pub fn new() -> Self {
        OrderBook {
            bids: BTreeMap::new(),
            asks: BTreeMap::new(),
        }
    }

    pub fn add_order(&mut self, order: Order) {
        match order.side {
            Side::Buy => {
                self.bids.entry(order.price).or_insert_with(Vec::new).push(order);
            },
            Side::Sell => {
                self.asks.entry(order.price).or_insert_with(Vec::new).push(order);
            }
        }
    }
}
