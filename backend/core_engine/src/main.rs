use tonic::{transport::Server, Request, Response, Status};
use tokio::sync::Mutex;
use std::sync::Arc;
use std::collections::HashMap;

// Import generated proto code
pub mod financial_entities {
    tonic::include_proto!("financial_entities");
}

use financial_entities::order_entry_server::{OrderEntry, OrderEntryServer};
use financial_entities::{OrderRequest, OrderResponse, OrderBookRequest, OrderBookStream, Order};

mod orderbook;
use orderbook::OrderBook;

#[derive(Debug, Default)]
pub struct FinancialOrderEntry {
    // In a real app, use a proper concurrent map or database
    books: Arc<Mutex<HashMap<String, OrderBook>>>,
}

#[tonic::async_trait]
impl OrderEntry for FinancialOrderEntry {
    async fn add_order(
        &self,
        request: Request<OrderRequest>,
    ) -> Result<Response<OrderResponse>, Status> {
        let req = request.into_inner();
        let symbol = req.symbol.clone();

        let mut books = self.books.lock().await;
        let book = books.entry(symbol.clone()).or_insert_with(|| OrderBook::new(&symbol));

        let order = orderbook::Order {
            id: format!("ord_{}", chrono::Utc::now().timestamp_nanos()),
            symbol: req.symbol,
            side: req.side,
            price: req.price,
            quantity: req.quantity,
            timestamp: chrono::Utc::now().timestamp(),
        };

        book.add_order(order.clone());

        let reply = OrderResponse {
            order_id: order.id,
            status: "ACCEPTED".into(),
            message: "Order added to book".into(),
        };

        Ok(Response::new(reply))
    }

    type GetOrderBookStream = tokio_stream::wrappers::ReceiverStream<Result<OrderBookStream, Status>>;

    async fn get_order_book(
        &self,
        request: Request<OrderBookRequest>,
    ) -> Result<Response<Self::GetOrderBookStream>, Status> {
        // Simple stream implementation placeholder
        let (tx, rx) = tokio::sync::mpsc::channel(4);

        tokio::spawn(async move {
            let update = OrderBookStream {
                bids: vec![],
                asks: vec![],
                timestamp: chrono::Utc::now().timestamp(),
            };
            tx.send(Ok(update)).await.unwrap();
        });

        Ok(Response::new(tokio_stream::wrappers::ReceiverStream::new(rx)))
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let addr = "[::]:50051".parse()?;
    let order_entry = FinancialOrderEntry::default();

    println!("Core Engine (Rust) listening on {}", addr);

    Server::builder()
        .add_service(OrderEntryServer::new(order_entry))
        .serve(addr)
        .await?;

    Ok(())
}
