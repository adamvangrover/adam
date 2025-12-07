-- Unified Ledger Schema for UFOS
-- Combines IB, WM, and AM domains

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- 1. WEALTH MANAGEMENT LAYER (WM)
CREATE TABLE portfolios (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    owner_name VARCHAR(255) NOT NULL,
    risk_profile JSONB, -- {gamma: 0.5, max_drawdown: 0.1, restrictions: ["TOBACCO"]}
    nav DECIMAL(20, 2) DEFAULT 0.00,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 2. ASSET MANAGEMENT LAYER (AM)
CREATE TABLE strategies (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    logic_path VARCHAR(512), -- Path to python/rust strategy code
    allocation_pct DECIMAL(5, 4), -- 0.0 to 1.0
    status VARCHAR(50) DEFAULT 'ACTIVE'
);

-- 3. INVESTMENT BANKING LAYER (IB)
CREATE TABLE inventory_ledger (
    asset_symbol VARCHAR(20) PRIMARY KEY,
    net_position DECIMAL(20, 8) DEFAULT 0,
    vwap_cost DECIMAL(20, 8) DEFAULT 0,
    unrealized_pnl DECIMAL(20, 2) DEFAULT 0,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 4. ORDER MANAGEMENT (Unified)
CREATE TYPE order_side AS ENUM ('BUY', 'SELL');
CREATE TYPE order_status AS ENUM ('NEW', 'WORKING', 'FILLED', 'CANCELLED', 'REJECTED');

CREATE TABLE orders (
    order_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    parent_id UUID, -- For strategy aggregation
    client_id UUID REFERENCES portfolios(id),
    strategy_id UUID REFERENCES strategies(id),

    symbol VARCHAR(20) NOT NULL,
    side order_side NOT NULL,
    quantity DECIMAL(20, 8) NOT NULL,
    price_limit DECIMAL(20, 8), -- NULL for Market

    status order_status DEFAULT 'NEW',
    filled_quantity DECIMAL(20, 8) DEFAULT 0,
    avg_fill_price DECIMAL(20, 8),

    internalization_flag BOOLEAN DEFAULT FALSE, -- True if filled against internal inventory

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 5. AI MEMORY LAYER
-- Uses pgvector (assumed installed)
CREATE TABLE memory_embeddings (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    content_chunk TEXT,
    embedding vector(768),
    metadata JSONB, -- {source: "email", sentiment: "negative", timestamp: ...}
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes
CREATE INDEX idx_orders_client ON orders(client_id);
CREATE INDEX idx_orders_status ON orders(status);
CREATE INDEX idx_memory_meta ON memory_embeddings USING GIN (metadata);
