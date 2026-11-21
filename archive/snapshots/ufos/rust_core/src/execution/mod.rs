pub mod orderbook;
pub mod matching_engine;

use serde::{Serialize, Deserialize};

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Side {
    Buy,
    Sell,
}

#[repr(C)]
#[repr(align(64))] // Cache line alignment
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Order {
    pub id: u64,
    pub price: u64, // Scaled integer
    pub quantity: u64,
    pub side: Side,
    pub timestamp: u64,
}
