
// 4.2 Resource Subscriptions

pub fn get_market_data_resource(symbol: &str) -> String {
    format!("financial://market/book/{}", symbol)
}
