use tonic::{transport::Server, Request, Response, Status};
use std::sync::{Arc, Mutex};
use std::collections::BTreeMap;
use std::collections::HashMap;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use std::time::Duration;

pub mod financial_entities {
    tonic::include_proto!("financial_entities");
}

use financial_entities::order_entry_server::{OrderEntry, OrderEntryServer};
use financial_entities::market_data_stream_server::{MarketDataStream, MarketDataStreamServer};
use financial_entities::{Order, OrderAck, SubscriptionRequest, Quote};

#[derive(Debug, Default)]
pub struct OrderBook {
    // Basic order book: symbol -> side ("BUY"|"SELL") -> price -> list of orders
    // Using BTreeMap for price ordering.
    // Buy orders: descending price priority.
    // Sell orders: ascending price priority.
    // For simplicity in this prototype, we'll just store all orders in a BTreeMap keyed by price.
    // In a real engine, we'd separate buy/sell and handle time priority.
    // Here: symbol -> BTreeMap<OrderedPrice, Vec<Order>>
    // Since BTreeMap keys must be Ord, and f64 isn't, we use a wrapper or just use ordered_float crate?
    // For zero-dependency simplicity in this scaffold, we'll stick to a simpler structure but use BTreeMap
    // to demonstrate intent.
    // We will use string representation of price for key to avoid float issues, or just a simple list for now
    // but the requirement is "BTreeMap".
    // Let's implement: symbol -> BTreeMap<u64, Vec<Order>> where u64 is price * 10000 (micros).
    books: Mutex<HashMap<String, BTreeMap<u64, Vec<Order>>>>,
}

impl OrderBook {
    fn add_order(&self, order: Order) {
        let mut books = self.books.lock().unwrap();
        let price_micros = (order.price * 10000.0) as u64;

        books.entry(order.symbol.clone())
            .or_insert_with(BTreeMap::new)
            .entry(price_micros)
            .or_insert_with(Vec::new)
            .push(order.clone());

        println!("Order added to book for {}: Price {}, Side {}", order.symbol, order.price, order.side);
    }
}

#[derive(Debug, Default)]
pub struct MyOrderEntry {
    order_book: Arc<OrderBook>,
}

#[tonic::async_trait]
impl OrderEntry for MyOrderEntry {
    async fn add_order(
        &self,
        request: Request<Order>,
    ) -> Result<Response<OrderAck>, Status> {
        let order = request.into_inner();
        println!("Received order: {:?}", order);

        self.order_book.add_order(order.clone());

        let reply = OrderAck {
            order_id: order.order_id,
            status: "ACCEPTED".into(),
            message: "Order received and added to book".into(),
        };

        Ok(Response::new(reply))
    }
}

#[derive(Debug, Default)]
pub struct MyMarketDataStream;

#[tonic::async_trait]
impl MarketDataStream for MyMarketDataStream {
    type SubscribeQuotesStream = ReceiverStream<Result<Quote, Status>>;

    async fn subscribe_quotes(
        &self,
        request: Request<SubscriptionRequest>,
    ) -> Result<Response<Self::SubscribeQuotesStream>, Status> {
        println!("Received subscription request: {:?}", request);
        let (tx, rx) = mpsc::channel(4);

        tokio::spawn(async move {
            let symbols = request.into_inner().symbols;
            // Mock streaming data
            loop {
                for symbol in &symbols {
                    let quote = Quote {
                        symbol: symbol.clone(),
                        bid: 100.0,
                        ask: 101.0,
                        bid_size: 10.0,
                        ask_size: 10.0,
                        timestamp: chrono::Utc::now().timestamp_millis(),
                    };
                    if let Err(_) = tx.send(Ok(quote)).await {
                        return; // Client disconnected
                    }
                }
                tokio::time::sleep(Duration::from_millis(1000)).await;
            }
        });

        Ok(Response::new(ReceiverStream::new(rx)))
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let addr = "[::1]:50051".parse()?;
    let order_book = Arc::new(OrderBook::default());
    let order_entry = MyOrderEntry { order_book: order_book.clone() };
    let market_data = MyMarketDataStream::default();

    println!("Core Engine Server listening on {}", addr);

    Server::builder()
        .add_service(OrderEntryServer::new(order_entry))
        .add_service(MarketDataStreamServer::new(market_data))
        .serve(addr)
        .await?;

    Ok(())
}
