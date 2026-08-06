# Dependency Map for ADAM Repository

This document outlines the dependencies across the modules in the ADAM repository.

## src/early_stage_valuation.py
- typing
- pydantic

## src/mcp_server.py
- typing
- logging
- src.core_valuation
- re
- core_valuation
- textblob
- core.engine.cyclical_reasoning_graph
- src.config
- json
- sys
- os
- config
- pydantic
- datetime
- mcp.server.fastmcp
- core.risk_engine.quantum_model
- pandas

## src/core_valuation.py
- typing
- config
- pandas

## src/credit_risk.py
- typing
- config

## src/agents/crypto_arbitrage.py
- typing
- pydantic

## src/agents/universal_arbitrage.py
- typing
- math
- pydantic

## src/llm/schemas.py
- typing
- pydantic

## src/llm/agent.py
- src.core.config
- pydantic_ai
- pydantic_ai.providers.litellm
- schemas

## src/api/main.py
- fastapi.middleware.cors
- routers
- fastapi

## src/api/routers/ingest.py
- typing
- json
- src.core.logging
- uuid
- os
- fastapi
- pydantic
- fastapi.responses
- src.orchestration.tasks
- asyncio

## src/adam/api/main.py
- src.adam.core.state_manager
- src.adam.core.optimizers
- src.adam.api.models
- contextlib
- fastapi
- torch
- src.adam.api.auth

## src/adam/api/auth.py
- typing
- fastapi.security
- os
- fastapi
- authlib.jose

## src/adam/api/models.py
- typing
- pydantic

## src/adam/core/state_manager.py
- logging
- typing
- pickle
- sys
- os
- core.security.safe_unpickler
- redis

## src/adam/core/optimizers.py
- torch.nn.functional
- torch
- torch.optim.optimizer
- math

## src/governance/gatekeeper.py
- typing
- urllib.error
- jsonschema
- json
- urllib.parse
- pydantic
- hashlib
- urllib.request
- asyncio

## src/core/logging.py
- logging
- sys

## src/core/config.py
- os
- pydantic_settings

## src/ingestion/plugin_manager.py
- typing
- src.core.logging
- sys
- os
- importlib
- base
- pkgutil

## src/ingestion/base.py
- typing
- io
- abc

## src/ingestion/plugins/pdf_parser.py
- typing
- src.ingestion.base
- fitz
- src.core.logging
- io

## src/ingestion/plugins/excel_parser.py
- typing
- src.core.logging
- base
- io
- pandas

## src/market_mayhem/scanners.py
- logging
- typing
- tenacity
- edgar
- pydantic
- pandas

## src/orchestration/celery_app.py
- src.core.config
- os
- celery

## src/orchestration/tasks.py
- typing
- celery.exceptions
- celery_app
- math
- src.llm.schemas
- src.ingestion.plugin_manager
- json
- src.core.logging
- celery
- src.llm.agent
- os
- redis
- src.core.config
- pandas

## src/schemas/core_types.py
- typing
- pydantic
- src.governance.gatekeeper
