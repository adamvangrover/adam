# Pragmatic v24 Migration Plan

This document outlines the migration to Adam v24.0, moving from a research prototype to a stable financial operating system.

## Overview
The v24 architecture introduces a hybrid runtime:
- **Rust Core Engine (`backend/core_engine`)**: Handles high-performance order matching and risk.
- **Python Intelligence Layer (`backend/intelligence`)**: Manages agents and RAG using Pydantic/Instructor for structured output.
- **Unified Ledger**: Uses TimescaleDB (time-series) and Qdrant (vectors) for data persistence.
- **Frontend (`services/webapp_v24`)**: A Next.js dashboard for real-time interaction.

## Directory Structure

```
/
├── backend/
│   ├── core_engine/       # Rust gRPC server
│   └── intelligence/      # Python agents & guardrails
├── shared/
│   └── proto/             # gRPC definitions
├── services/
│   ├── webapp_v24/        # Next.js Frontend
│   └── webapp/            # Legacy Flask/React app
├── core/                  # Legacy v23 Core (Functionality preserved)
└── docker-compose.yml     # Updated infrastructure
```

## Running the System

1. **Start Infrastructure**:
   ```bash
   docker-compose up -d timescaledb qdrant redis
   ```

2. **Run Rust Core**:
   ```bash
   cd backend/core_engine
   cargo run
   ```

3. **Run Next.js Frontend**:
   ```bash
   cd services/webapp_v24
   npm install
   npm run dev
   ```

## Next Steps (Phases 3 & 4)
- Connect Frontend to Rust Core via WebSocket/gRPC-Web.
- Implement full Agent workflow in `backend/intelligence`.
- Migrate data from `data/omni_graph` to Qdrant/TimescaleDB.
