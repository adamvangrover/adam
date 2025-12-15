use uuid::Uuid;

#[derive(Debug, Clone)]
pub enum Side {
    Buy,
    Sell,
}

#[derive(Debug, Clone)]
pub struct Order {
    pub order_id: Uuid,
    pub parent_id: Option<Uuid>,
    pub client_id: String,
    pub desk_id: String,
    pub strategy_tag: String,
    pub side: Side,
    pub internalization_flag: bool,
    pub price: f64,
    pub quantity: f64,
}

pub struct UnifiedLedger {
    pub orders: Vec<Order>,
}

impl UnifiedLedger {
    pub fn new() -> Self {
        UnifiedLedger { orders: Vec::new() }
    }

    pub fn add_order(&mut self, order: Order) {
        self.orders.push(order);
    }
}
