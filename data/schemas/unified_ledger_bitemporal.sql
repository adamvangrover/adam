-- Adam v24.0 Unified Ledger Schema
-- Table 1: Unified Ledger Order Schema Hierarchy

CREATE TYPE intent_side_enum AS ENUM ('Buy', 'Sell');

CREATE TABLE unified_ledger (
    order_id UUID PRIMARY KEY,
    parent_id UUID, -- Reference to aggregate strategy order (AM)
    client_id VARCHAR(255) NOT NULL, -- Identifier for ultimate beneficiary (WM)
    desk_id VARCHAR(255) NOT NULL, -- Identifier for trading desk (IB)
    strategy_tag VARCHAR(255), -- Algo strategy ID (AM/IB)
    intent_side intent_side_enum NOT NULL,
    symbol VARCHAR(50) NOT NULL,
    price DECIMAL(18, 8),
    quantity DECIMAL(18, 8),
    internalization_flag BOOLEAN DEFAULT FALSE, -- True if filled against internal inventory

    -- Bitemporal Columns (Chronos)
    valid_time_start TIMESTAMP WITH TIME ZONE NOT NULL,
    valid_time_end TIMESTAMP WITH TIME ZONE,
    transaction_time_start TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    transaction_time_end TIMESTAMP WITH TIME ZONE,

    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_unified_ledger_parent ON unified_ledger(parent_id);
CREATE INDEX idx_unified_ledger_client ON unified_ledger(client_id);
CREATE INDEX idx_unified_ledger_symbol ON unified_ledger(symbol);
CREATE INDEX idx_unified_ledger_valid_time ON unified_ledger(valid_time_start, valid_time_end);

-- Contextual Layer (Vector References)
CREATE TABLE contextual_artifacts (
    artifact_id UUID PRIMARY KEY,
    content_vector VECTOR(1536), -- Assuming OpenAI embeddings
    source_type VARCHAR(50), -- e.g., 'News', 'Email', 'AnalystReport'
    content_text TEXT,
    associated_order_id UUID REFERENCES unified_ledger(order_id)
);
