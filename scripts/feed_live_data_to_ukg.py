
import sys
import os
import json
import networkx as nx
from datetime import datetime

# Add root to pythonpath
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.data_sources.data_fetcher import DataFetcher
from core.engine.unified_knowledge_graph import UnifiedKnowledgeGraph
from core.utils.logging_utils import get_logger

logger = get_logger(__name__)

def main():
    logger.info("Initializing DataFetcher and Knowledge Graph...")
    fetcher = DataFetcher()
    ukg = UnifiedKnowledgeGraph()

    # --- Fetch Indices ---
    indices = {
        "^GSPC": "S&P 500",
        "^DJI": "Dow Jones",
        "^IXIC": "Nasdaq Composite",
        "BTC-USD": "Bitcoin",
        "BZ=F": "Brent Crude Oil",
        "GC=F": "Gold",
        "^TNX": "10-Year Treasury Yield"
    }

    logger.info("Fetching Market Indices...")
    for symbol, name in indices.items():
        try:
            # Get current price
            snap = fetcher.fetch_realtime_snapshot(symbol)
            current_price = snap.get("last_price")

            # Get history for WoW (approx 5 trading days = 1 week)
            hist = fetcher.fetch_historical_data(symbol, period="5d", interval="1d")
            wow_change = 0.0
            if hist and len(hist) > 0 and current_price:
                # 5d returns 5 days. Oldest is ~1 week ago.
                prev_price = hist[0].get("close")
                if prev_price:
                    wow_change = (current_price - prev_price) / prev_price

            # Add to UKG
            node_id = f"Index:{symbol}"
            ukg.graph.add_node(node_id,
                type="MarketIndex",
                symbol=symbol,
                name=name,
                price=current_price,
                wow_change=wow_change,
                timestamp=datetime.now().isoformat()
            )
            logger.info(f"Ingested Index: {name} (${current_price} | {wow_change:.2%})")

        except Exception as e:
            logger.error(f"Failed to fetch index {name}: {e}")

    # --- Fetch Companies ---
    tickers = ["AAPL", "NVDA", "TSLA", "MSFT", "GOOGL", "AMZN", "META"]

    logger.info(f"Fetching live data for: {tickers}")

    for ticker in tickers:
        try:
            # 1. Fetch Data
            data = fetcher.fetch_market_data(ticker)
            if not data:
                logger.warning(f"Skipping {ticker}: No data found.")
                continue

            company_name = data.get("description", "").split(".")[0] if data.get("description") else f"{ticker} Corp"
            # Cleanup name
            if "Inc" in company_name: company_name = company_name.replace("Inc", "").strip()

            # 2. Update/Create Company Node
            # We use the Ticker as the primary ID for simplicity in this script,
            # or try to map to existing names in UKG.
            # UKG seed uses names like "Apple Inc.".

            node_id = f"Company:{ticker}"

            if node_id not in ukg.graph:
                ukg.graph.add_node(node_id, type="Company", ticker=ticker, name=company_name)
                # Link to generic Company concept
                ukg.graph.add_edge(node_id, "Company", relation="is_a", type="fibo")

            # Update static properties
            ukg.graph.nodes[node_id].update({
                "sector": data.get("sector"),
                "industry": data.get("industry"),
                "description": data.get("description"),
                "last_updated": datetime.now().isoformat()
            })

            # 3. Create MarketSnapshot Node (Time-series data)
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            snapshot_id = f"Snapshot:{ticker}:{timestamp}"

            snapshot_attrs = {
                "type": "MarketSnapshot",
                "ticker": ticker,
                "price": data.get("current_price"),
                "pe_ratio": data.get("pe_ratio"),
                "market_cap": data.get("market_cap"),
                "volume": data.get("volume"),
                "timestamp": timestamp,
                "provenance": "yfinance_api"
            }

            ukg.graph.add_node(snapshot_id, **snapshot_attrs)
            ukg.graph.add_edge(node_id, snapshot_id, relation="has_snapshot", type="temporal")

            # 3b. Fetch and Ingest News
            news_items = fetcher.fetch_news(ticker)
            for idx, item in enumerate(news_items[:3]): # Limit to top 3
                title = item.get("title")
                if not title: continue

                # Create News Node
                # Use a hash or sanitized title as ID
                news_id = f"News:{ticker}:{idx}"

                news_attrs = {
                    "type": "NewsArticle",
                    "title": title,
                    "publisher": item.get("publisher"),
                    "link": item.get("link"),
                    "timestamp": datetime.now().isoformat(), # yfinance news date is messy, use ingest time or try parse
                    "provenance": "yfinance_news"
                }

                if news_id not in ukg.graph:
                    ukg.graph.add_node(news_id, **news_attrs)
                    ukg.graph.add_edge(node_id, news_id, relation="mentioned_in", type="informational")
                    ukg.graph.add_edge(news_id, node_id, relation="mentions", type="informational")

            # 4. Link to Sector
            sector = data.get("sector")
            if sector:
                sector_node = f"Sector:{sector}"
                if sector_node not in ukg.graph:
                    ukg.graph.add_node(sector_node, type="MarketSector", name=sector)
                    ukg.graph.add_edge(sector_node, "MarketSector", relation="is_a", type="fibo")

                ukg.graph.add_edge(node_id, sector_node, relation="belongs_to", type="fibo")

            logger.info(f"Successfully ingested {ticker} into Knowledge Graph.")

        except Exception as e:
            logger.error(f"Error processing {ticker}: {e}")

    # 5. Save the Graph
    output_dir = "data/snapshots"
    os.makedirs(output_dir, exist_ok=True)

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"ukg_snapshot_{timestamp_str}.graphml"
    filepath = os.path.join(output_dir, filename)

    logger.info(f"Saving Knowledge Graph snapshot to {filepath}...")

    # 1. Save as JSON (Robust)
    try:
        json_filepath = filepath.replace(".graphml", ".json")
        data = nx.node_link_data(ukg.graph)
        with open(json_filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        logger.info(f"JSON snapshot saved to {json_filepath}")
    except Exception as e:
        logger.error(f"Failed to save JSON snapshot: {e}")

    # 2. Save as GraphML (Requires flattening)
    try:
        # Create a copy to flatten attributes
        G_flat = ukg.graph.copy()
        for node, attrs in G_flat.nodes(data=True):
            for k, v in attrs.items():
                if isinstance(v, (dict, list)):
                    attrs[k] = json.dumps(v)  # Serialize complex types to string
                elif v is None:
                    attrs[k] = ""

        # Flatten edge attributes too if needed
        if G_flat.is_multigraph():
            for u, v, k, attrs in G_flat.edges(data=True, keys=True):
                for key, val in attrs.items():
                    if isinstance(val, (dict, list)):
                        attrs[key] = json.dumps(val)
        else:
             for u, v, attrs in G_flat.edges(data=True):
                for key, val in attrs.items():
                    if isinstance(val, (dict, list)):
                        attrs[key] = json.dumps(val)

        nx.write_graphml(G_flat, filepath)
        logger.info(f"GraphML snapshot saved to {filepath}")

    except Exception as e:
        logger.error(f"Failed to save GraphML snapshot: {e}")

if __name__ == "__main__":
    main()
