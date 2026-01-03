-- Unified Ledger Schema
-- Links Strategy (Parent) to Execution (Child) orders

CREATE TABLE IF NOT EXISTS strategies (
    strategy_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS parent_orders (
    parent_order_id TEXT PRIMARY KEY,
    strategy_id TEXT REFERENCES strategies(strategy_id),
    symbol TEXT NOT NULL,
    side TEXT NOT NULL, -- BUY, SELL
    target_quantity DECIMAL NOT NULL,
    algo_type TEXT NOT NULL, -- TWAP, VWAP, POV
    start_time TIMESTAMPTZ NOT NULL,
    end_time TIMESTAMPTZ,
    status TEXT NOT NULL, -- ACTIVE, COMPLETED, CANCELLED
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS child_orders (
    child_order_id TEXT PRIMARY KEY,
    parent_order_id TEXT REFERENCES parent_orders(parent_order_id),
    exchange_order_id TEXT,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    price DECIMAL,
    quantity DECIMAL NOT NULL,
    filled_quantity DECIMAL DEFAULT 0,
    status TEXT NOT NULL, -- NEW, PARTIALLY_FILLED, FILLED, CANCELED
    venue TEXT NOT NULL, -- NYSE, NASDAQ, IEX
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS executions (
    execution_id TEXT PRIMARY KEY,
    child_order_id TEXT REFERENCES child_orders(child_order_id),
    price DECIMAL NOT NULL,
    quantity DECIMAL NOT NULL,
    liquidity_flag TEXT, -- MAKER, TAKER
    execution_time TIMESTAMPTZ NOT NULL
);

-- TimescaleDB Hypertable for high-frequency market data
CREATE TABLE IF NOT EXISTS market_data (
    time TIMESTAMPTZ NOT NULL,
    symbol TEXT NOT NULL,
    bid_price DECIMAL,
    ask_price DECIMAL,
    bid_size DECIMAL,
    ask_size DECIMAL
);

SELECT create_hypertable('market_data', 'time', if_not_exists => TRUE);
