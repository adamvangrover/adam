use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::cmp::Ordering;
use std::sync::{Arc, Mutex};

#[derive(Clone, Copy, PartialEq, PartialOrd)]
pub struct OrderedFloat(pub f64);

impl Eq for OrderedFloat {}

impl Ord for OrderedFloat {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

#[derive(Clone, Debug)]
pub struct RustOrder {
    pub order_id: String,
    pub symbol: String,
    pub side: String,
    pub quantity: f64,
    pub price: Option<f64>,
    pub order_type: String,
    pub status: String,
    pub filled_quantity: f64,
}

type OrderRef = Arc<Mutex<RustOrder>>;

pub struct RustOrderBook {
    pub symbol: String,
    pub bids_levels: BTreeMap<OrderedFloat, VecDeque<OrderRef>>,
    pub asks_levels: BTreeMap<OrderedFloat, VecDeque<OrderRef>>,
    pub order_map: HashMap<String, OrderRef>,
}

impl RustOrderBook {
    pub fn new(symbol: String) -> Self {
        Self {
            symbol,
            bids_levels: BTreeMap::new(),
            asks_levels: BTreeMap::new(),
            order_map: HashMap::new(),
        }
    }

    pub fn cancel_order(&mut self, order_id: &str) -> bool {
        if let Some(order) = self.order_map.remove(order_id) {
            order.lock().unwrap().status = "CANCELED".to_string();
            return true;
        }
        false
    }

    pub fn add_order(&mut self, mut incoming_order: RustOrder) -> Vec<(OrderRef, f64)> {
        let mut fills = Vec::new();
        let mut remaining_qty = incoming_order.quantity;

        if incoming_order.side == "BUY" {
            let mut prices_to_remove = Vec::new();

            for (&price_key, level_queue) in self.asks_levels.iter_mut() {
                if remaining_qty <= 0.0 {
                    break;
                }

                if incoming_order.order_type == "LIMIT" {
                    if let Some(limit_price) = incoming_order.price {
                        if price_key.0 > limit_price {
                            break;
                        }
                    }
                }

                let mut orders_to_remove = 0;
                for best_ask_ref in level_queue.iter_mut() {
                    if remaining_qty <= 0.0 {
                        break;
                    }

                    let mut best_ask_order = best_ask_ref.lock().unwrap();
                    if best_ask_order.status == "CANCELED" {
                        orders_to_remove += 1;
                        continue;
                    }

                    let available_qty = best_ask_order.quantity - best_ask_order.filled_quantity;
                    let match_qty = if remaining_qty < available_qty { remaining_qty } else { available_qty };

                    fills.push((Arc::clone(best_ask_ref), match_qty));

                    best_ask_order.filled_quantity += match_qty;
                    incoming_order.filled_quantity += match_qty;
                    remaining_qty -= match_qty;

                    if best_ask_order.filled_quantity >= best_ask_order.quantity {
                        orders_to_remove += 1;
                        self.order_map.remove(&best_ask_order.order_id);
                    }
                }

                for _ in 0..orders_to_remove {
                    level_queue.pop_front();
                }
                if level_queue.is_empty() {
                    prices_to_remove.push(price_key);
                }
            }

            for p in prices_to_remove {
                self.asks_levels.remove(&p);
            }

            if remaining_qty > 0.0 && incoming_order.order_type == "LIMIT" {
                self._add_bid(incoming_order);
            }

        } else if incoming_order.side == "SELL" {
            let mut prices_to_remove = Vec::new();

            for (&price_key, level_queue) in self.bids_levels.iter_mut().rev() {
                if remaining_qty <= 0.0 {
                    break;
                }

                if incoming_order.order_type == "LIMIT" {
                    if let Some(limit_price) = incoming_order.price {
                        if price_key.0 < limit_price {
                            break;
                        }
                    }
                }

                let mut orders_to_remove = 0;
                for best_bid_ref in level_queue.iter_mut() {
                    if remaining_qty <= 0.0 {
                        break;
                    }

                    let mut best_bid_order = best_bid_ref.lock().unwrap();
                    if best_bid_order.status == "CANCELED" {
                        orders_to_remove += 1;
                        continue;
                    }

                    let available_qty = best_bid_order.quantity - best_bid_order.filled_quantity;
                    let match_qty = if remaining_qty < available_qty { remaining_qty } else { available_qty };

                    fills.push((Arc::clone(best_bid_ref), match_qty));

                    best_bid_order.filled_quantity += match_qty;
                    incoming_order.filled_quantity += match_qty;
                    remaining_qty -= match_qty;

                    if best_bid_order.filled_quantity >= best_bid_order.quantity {
                        orders_to_remove += 1;
                        self.order_map.remove(&best_bid_order.order_id);
                    }
                }

                for _ in 0..orders_to_remove {
                    level_queue.pop_front();
                }
                if level_queue.is_empty() {
                    prices_to_remove.push(price_key);
                }
            }

            for p in prices_to_remove {
                self.bids_levels.remove(&p);
            }

            if remaining_qty > 0.0 && incoming_order.order_type == "LIMIT" {
                self._add_ask(incoming_order);
            }
        }

        fills
    }

    fn _add_bid(&mut self, order: RustOrder) {
        if let Some(price) = order.price {
            let order_id = order.order_id.clone();
            let order_ref = Arc::new(Mutex::new(order));
            self.bids_levels.entry(OrderedFloat(price)).or_insert_with(VecDeque::new).push_back(Arc::clone(&order_ref));
            self.order_map.insert(order_id, order_ref);
        }
    }

    fn _add_ask(&mut self, order: RustOrder) {
        if let Some(price) = order.price {
            let order_id = order.order_id.clone();
            let order_ref = Arc::new(Mutex::new(order));
            self.asks_levels.entry(OrderedFloat(price)).or_insert_with(VecDeque::new).push_back(Arc::clone(&order_ref));
            self.order_map.insert(order_id, order_ref);
        }
    }

    pub fn get_l2_snapshot(&mut self, py: Python, depth: usize) -> PyResult<PyObject> {
        let bids = PyList::empty(py);
        let mut count = 0;

        for (&price_key, level_queue) in self.bids_levels.iter().rev() {
            if count >= depth { break; }
            let mut qty = 0.0;
            for order_ref in level_queue.iter() {
                let order = order_ref.lock().unwrap();
                if order.status != "CANCELED" {
                    qty += order.quantity - order.filled_quantity;
                }
            }
            if qty > 0.0 {
                let tuple = pyo3::types::PyTuple::new(py, &[price_key.0, qty]);
                bids.append(tuple)?;
                count += 1;
            }
        }

        let asks = PyList::empty(py);
        count = 0;
        for (&price_key, level_queue) in self.asks_levels.iter() {
            if count >= depth { break; }
            let mut qty = 0.0;
            for order_ref in level_queue.iter() {
                let order = order_ref.lock().unwrap();
                if order.status != "CANCELED" {
                    qty += order.quantity - order.filled_quantity;
                }
            }
            if qty > 0.0 {
                let tuple = pyo3::types::PyTuple::new(py, &[price_key.0, qty]);
                asks.append(tuple)?;
                count += 1;
            }
        }

        let result = PyDict::new(py);
        result.set_item("symbol", &self.symbol)?;
        result.set_item("bids", bids)?;
        result.set_item("asks", asks)?;

        Ok(result.into())
    }
}

#[pyclass]
pub struct RustMatchingEngine {
    books: HashMap<String, RustOrderBook>,
}

#[pymethods]
impl RustMatchingEngine {
    #[new]
    pub fn new() -> Self {
        Self {
            books: HashMap::new(),
        }
    }

    pub fn cancel_order(&mut self, symbol: &str, order_id: &str) -> bool {
        if let Some(book) = self.books.get_mut(symbol) {
            return book.cancel_order(order_id);
        }
        false
    }

    pub fn process_order<'py>(&mut self, py: Python<'py>, order_dict: &'py PyDict) -> PyResult<&'py PyDict> {
        let order_id: String = order_dict.get_item("order_id").unwrap().extract()?;
        let symbol: String = order_dict.get_item("symbol").unwrap().extract()?;
        let side: String = order_dict.get_item("side").unwrap().extract()?;
        let quantity: f64 = order_dict.get_item("quantity").unwrap().extract()?;

        let price_item = order_dict.get_item("price");
        let price: Option<f64> = match price_item {
            Some(p) => if p.is_none() { None } else { Some(p.extract()?) },
            None => None,
        };

        let order_type: String = order_dict.get_item("order_type").unwrap().extract()?;

        let status_item = order_dict.get_item("status");
        let status: String = if let Some(s) = status_item { s.extract()? } else { "PENDING".to_string() };

        let fq_item = order_dict.get_item("filled_quantity");
        let filled_quantity: f64 = if let Some(fq) = fq_item { fq.extract()? } else { 0.0 };

        let mut incoming_order = RustOrder {
            order_id: order_id.clone(),
            symbol: symbol.clone(),
            side,
            quantity,
            price,
            order_type,
            status,
            filled_quantity,
        };

        let book = self.books.entry(symbol.clone()).or_insert_with(|| RustOrderBook::new(symbol.clone()));

        let fills = book.add_order(incoming_order.clone());

        let mut total_fill_cost = 0.0;
        let mut total_fill_qty = 0.0;
        let processed_fills = PyList::empty(py);

        let datetime_module = py.import("datetime")?;
        let datetime_class = datetime_module.getattr("datetime")?;

        for (matched_ref, match_qty) in fills {
            let matched_order = matched_ref.lock().unwrap();
            let matched_price = matched_order.price.unwrap_or(0.0);

            let cost = matched_price * match_qty;
            total_fill_cost += cost;
            total_fill_qty += match_qty;

            let fill_dict = PyDict::new(py);
            fill_dict.set_item("maker_order_id", &matched_order.order_id)?;
            fill_dict.set_item("taker_order_id", &order_id)?;
            fill_dict.set_item("price", matched_price)?;
            fill_dict.set_item("quantity", match_qty)?;

            let utcnow = datetime_class.getattr("utcnow")?.call0()?;
            let isoformat = utcnow.getattr("isoformat")?.call0()?;
            fill_dict.set_item("timestamp", isoformat)?;

            processed_fills.append(fill_dict)?;
        }

        if total_fill_qty > 0.0 {
            incoming_order.filled_quantity += total_fill_qty;
        }

        let new_status = if incoming_order.filled_quantity >= incoming_order.quantity {
            "FILLED"
        } else {
            if incoming_order.filled_quantity > 0.0 { "PARTIALLY_FILLED" } else { "PENDING" }
        };

        let result_dict = PyDict::new(py);
        result_dict.set_item("order_id", order_id)?;
        result_dict.set_item("symbol", symbol)?;
        result_dict.set_item("status", new_status)?;
        result_dict.set_item("filled_quantity", incoming_order.filled_quantity)?;
        result_dict.set_item("fills", processed_fills)?;

        if total_fill_qty > 0.0 {
            let avg_price = total_fill_cost / total_fill_qty;
            result_dict.set_item("average_fill_price", avg_price)?;
        }

        Ok(result_dict)
    }

    pub fn get_l2_snapshot(&mut self, py: Python, symbol: &str, depth: usize) -> PyResult<PyObject> {
        if let Some(book) = self.books.get_mut(symbol) {
            book.get_l2_snapshot(py, depth)
        } else {
            let result = PyDict::new(py);
            result.set_item("symbol", symbol)?;
            result.set_item("bids", PyList::empty(py))?;
            result.set_item("asks", PyList::empty(py))?;
            Ok(result.into())
        }
    }
}
