use tonic::{transport::Server, Request, Response, Status};
use tokio::sync::{RwLock, mpsc};
use tokio_stream::wrappers::ReceiverStream;
use std::sync::Arc;
use std::collections::{HashMap, BTreeMap};
use std::time::Duration;
use chrono::Utc;

// -----------------------------------------------------------------------------
// PROTO MODULE
// -----------------------------------------------------------------------------
pub mod financial_entities {
    tonic::include_proto!("financial_entities");
}

// Import all generated structs
use financial_entities::{
    order_entry_server::{OrderEntry, OrderEntryServer},
    market_data_stream_server::{MarketDataStream, MarketDataStreamServer},
    Order, OrderAck, SubscriptionRequest, Quote, OrderBookRequest, OrderBookStream
};

// -----------------------------------------------------------------------------
// CORE ENGINE: ORDER BOOK LOGIC
// -----------------------------------------------------------------------------

/// precise_price converts a float price to micro-units (u64) to avoid 
/// floating point comparison issues in BTreeMaps.
fn precise_price(price: f64) -> u64 {
    (price * 1_000_000.0).round() as u64
}

#[derive(Debug, Default)]
struct SecurityBook {
    /// Bids: Buy orders, ordered High -> Low (Reverse)
    /// We use u64 (micros) as key. BTreeMap sorts Low -> High by default, 
    /// so we will iterate using `.rev()` for Bids.
    bids: BTreeMap<u64, Vec<Order>>,
    
    /// Asks: Sell orders, ordered Low -> High
    asks: BTreeMap<u64, Vec<Order>>,
}

impl SecurityBook {
    fn add(&mut self, order: Order) {
        let price_key = precise_price(order.price);
        match order.side.as_str() {
            "BUY" => {
                self.bids.entry(price_key).or_default().push(order);
            }
            "SELL" => {
                self.asks.entry(price_key).or_default().push(order);
            }
            _ => eprintln!("Invalid side: {}", order.side),
        }
    }

    /// Generates a snapshot of the top N levels for streaming
    fn snapshot(&self, symbol: &str) -> OrderBookStream {
        // Flatten the maps to lists of orders for the proto response
        // In a real scenario, you would aggregate volume at price levels here.
        let bids = self.bids.iter().rev().take(10)
            .flat_map(|(_, orders)| orders.clone())
            .collect();
            
        let asks = self.asks.iter().take(10)
            .flat_map(|(_, orders)| orders.clone())
            .collect();

        OrderBookStream {
            symbol: symbol.to_string(),
            bids,
            asks,
            timestamp: Utc::now().timestamp_millis(),
        }
    }
}

#[derive(Debug, Default)]
pub struct EngineState {
    // RwLock allows multiple readers (Streamers) to access simultaneously.
    // Only locks exclusively during Order Entry.
    books: RwLock<HashMap<String, SecurityBook>>,
}

// -----------------------------------------------------------------------------
// SERVICE 1: ORDER ENTRY
// -----------------------------------------------------------------------------
#[derive(Debug)]
pub struct FinancialOrderEntry {
    state: Arc<EngineState>,
}

#[tonic::async_trait]
impl OrderEntry for FinancialOrderEntry {
    async fn add_order(
        &self,
        request: Request<Order>,
    ) -> Result<Response<OrderAck>, Status> {
        let mut order = request.into_inner();
        
        // Validation
        if order.price <= 0.0 || order.quantity <= 0.0 {
            return Err(Status::invalid_argument("Price and Quantity must be positive"));
        }

        // Augment with server-side metadata
        order.timestamp = Utc::now().timestamp_millis();
        let symbol = order.symbol.clone();
        let order_id = order.order_id.clone();

        // Lock for Write
        let mut books = self.state.books.write().await;
        let book = books.entry(symbol.clone()).or_insert_with(SecurityBook::default);
        
        book.add(order);
        println!("INFO: Order {} accepted for {}", order_id, symbol);

        Ok(Response::new(OrderAck {
            order_id,
            status: "ACCEPTED".into(),
            message: "Order successfully committed to engine memory".into(),
            timestamp: Utc::now().timestamp_millis(),
        }))
    }

    // New method implementation from 'main' branch requirements
    async fn get_order_book_snapshot(
        &self,
        request: Request<OrderBookRequest>,
    ) -> Result<Response<OrderBookStream>, Status> {
        let req = request.into_inner();
        let books = self.state.books.read().await;

        if let Some(book) = books.get(&req.symbol) {
            Ok(Response::new(book.snapshot(&req.symbol)))
        } else {
            Err(Status::not_found("Symbol not found in engine"))
        }
    }
}

// -----------------------------------------------------------------------------
// SERVICE 2: MARKET DATA STREAM
// -----------------------------------------------------------------------------
#[derive(Debug)]
pub struct FinancialMarketData {
    state: Arc<EngineState>,
}

#[tonic::async_trait]
impl MarketDataStream for FinancialMarketData {
    type SubscribeQuotesStream = ReceiverStream<Result<Quote, Status>>;

    async fn subscribe_quotes(
        &self,
        request: Request<SubscriptionRequest>,
    ) -> Result<Response<Self::SubscribeQuotesStream>, Status> {
        let req = request.into_inner();
        println!("INFO: Client subscribed to Market Data: {:?}", req.symbols);

        let (tx, rx) = mpsc::channel(16);
        
        // Spawn a task to simulate streaming updates
        // In production, this would subscribe to an internal Event Bus
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_millis(500));
            loop {
                interval.tick().await; // Wait for tick

                for sym in &req.symbols {
                    // Mock data generation (replace with real book listeners later)
                    let quote = Quote {
                        symbol: sym.clone(),
                        bid: 100.0 + (rand::random::<f64>() * 2.0),
                        ask: 102.0 + (rand::random::<f64>() * 2.0),
                        bid_size: 500.0,
                        ask_size: 500.0,
                        timestamp: Utc::now().timestamp_millis(),
                    };

                    if tx.send(Ok(quote)).await.is_err() {
                        println!("WARN: Client disconnected, stopping stream for {:?}", req.symbols);
                        return;
                    }
                }
            }
        });

        Ok(Response::new(ReceiverStream::new(rx)))
    }
}

// -----------------------------------------------------------------------------
// MAIN ENTRY POINT
// -----------------------------------------------------------------------------
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let addr = "[::1]:50051".parse()?;
    
    // Shared State Container
    let shared_state = Arc::new(EngineState::default());

    // Service Initialization
    let order_entry_svc = FinancialOrderEntry { 
        state: shared_state.clone() 
    };
    let market_data_svc = FinancialMarketData { 
        state: shared_state.clone() 
    };

    println!("Core Engine v0.1.0 listening on {}", addr);
    println!("├── Service: OrderEntry (Active)");
    println!("└── Service: MarketDataStream (Active)");

    Server::builder()
        .add_service(OrderEntryServer::new(order_entry_svc))
        .add_service(MarketDataStreamServer::new(market_data_svc))
        .serve(addr)
        .await?;

    Ok(())
}