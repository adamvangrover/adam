# Outstanding Errors and Issues

## Known Issues (FO Super-App Integration)

1.  **Live Connections:** While `core/market_data` now supports `yfinance` and `core/pricing_engine` uses realistic GBM simulation, connection to institutional feeds (Bloomberg, Fix Protocol) is still pending implementation.
2.  **Vector Search:** `core/memory/engine.py` persists to `data/personal_memory.db` but relies on keyword search. True semantic search (Chroma/FAISS) requires integrating the `embeddings` module.
3.  **UI Integration:** The backend logic is fully integrated into `MetaOrchestrator` (including Family Office routes), but the frontend `showcase/` dashboard does not yet expose specific widgets for the FO Super-App.
4.  **Dependencies:** The system now requires `langgraph`, `numpy`, `pandas`, `transformers`, `torch`, `spacy`, `textblob`, `tweepy`, `scikit-learn`, `beautifulsoup4`, `redis`, `pika`, `python-dotenv`, `tiktoken`, `semantic-kernel`, and `langchain-community`.
