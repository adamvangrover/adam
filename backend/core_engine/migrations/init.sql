-- Initial schema for Unified Ledger

CREATE TABLE IF NOT EXISTS orders (
    order_id TEXT PRIMARY KEY,
    parent_order_id TEXT, -- Link to Strategy
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    price DECIMAL NOT NULL,
    quantity DECIMAL NOT NULL,
    order_type TEXT NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL
);

CREATE TABLE IF NOT EXISTS trades (
    trade_id TEXT PRIMARY KEY,
    order_id TEXT NOT NULL REFERENCES orders(order_id),
    symbol TEXT NOT NULL,
    price DECIMAL NOT NULL,
    quantity DECIMAL NOT NULL,
    side TEXT NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL
);

-- Hypertable for trades (TimescaleDB)
-- SELECT create_hypertable('trades', 'timestamp');
