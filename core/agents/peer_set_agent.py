from typing import Any, Dict, List, Union, Optional
import logging
import asyncio
from core.agents.agent_base import AgentBase
from core.schemas.agent_schema import AgentInput, AgentOutput

# Data Science imports
try:
    import yfinance as yf
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    yf = None
    TfidfVectorizer = None
    cosine_similarity = None

logger = logging.getLogger(__name__)

class PeerSetAgent(AgentBase):
    """
    Agent responsible for building peer sets based on industry classification codes
    (GICS, NAICS, NACE) and product/market overlaps using real-time data and semantic analysis.
    """

    # Comprehensive Mock Database (Fallback & Candidate Pool)
    COMPANY_DB = [
        # Technology - Hardware
        {
            "ticker": "AAPL", "name": "Apple Inc.",
            "sector": "Information Technology", "industry": "Technology Hardware, Storage & Peripherals",
            "gics": "45202010", "naics": "334111", "nace": "26.20",
            "products": ["Smartphones", "Laptops", "Tablets", "Wearables", "Services"],
            "markets": ["Global", "Consumer Electronics", "Enterprise"]
        },
        {
            "ticker": "DELL", "name": "Dell Technologies Inc.",
            "sector": "Information Technology", "industry": "Technology Hardware, Storage & Peripherals",
            "gics": "45202010", "naics": "334111", "nace": "26.20",
            "products": ["Laptops", "Servers", "Storage", "Peripherals"],
            "markets": ["Global", "Enterprise", "Consumer"]
        },
        {
            "ticker": "HPQ", "name": "HP Inc.",
            "sector": "Information Technology", "industry": "Technology Hardware, Storage & Peripherals",
            "gics": "45202010", "naics": "334111", "nace": "26.20",
            "products": ["Laptops", "Printers", "3D Printing"],
            "markets": ["Global", "Consumer", "Enterprise"]
        },
        # Technology - Software & Services
        {
            "ticker": "MSFT", "name": "Microsoft Corporation",
            "sector": "Information Technology", "industry": "Software",
            "gics": "45102010", "naics": "511210", "nace": "58.29",
            "products": ["Operating Systems", "Cloud Computing", "Productivity Software", "Gaming"],
            "markets": ["Global", "Enterprise", "Consumer", "Government"]
        },
        {
            "ticker": "ORCL", "name": "Oracle Corporation",
            "sector": "Information Technology", "industry": "Software",
            "gics": "45102010", "naics": "511210", "nace": "58.29",
            "products": ["Database Software", "Cloud Infrastructure", "Enterprise Applications"],
            "markets": ["Global", "Enterprise"]
        },
        {
            "ticker": "ADBE", "name": "Adobe Inc.",
            "sector": "Information Technology", "industry": "Software",
            "gics": "45102010", "naics": "511210", "nace": "58.29",
            "products": ["Creative Software", "Digital Experience", "Document Cloud"],
            "markets": ["Global", "Creative Professionals", "Enterprise"]
        },
        {
            "ticker": "CRM", "name": "Salesforce, Inc.",
            "sector": "Information Technology", "industry": "Software",
            "gics": "45102010", "naics": "511210", "nace": "58.29",
            "products": ["CRM", "Cloud Applications", "Enterprise Software"],
            "markets": ["Global", "Enterprise"]
        },
        # Communication Services
        {
            "ticker": "GOOGL", "name": "Alphabet Inc.",
            "sector": "Communication Services", "industry": "Interactive Media & Services",
            "gics": "50201010", "naics": "518210", "nace": "63.11",
            "products": ["Search Engine", "Online Advertising", "Cloud Computing", "Mobile OS", "Video Streaming"],
            "markets": ["Global", "Digital Advertising", "Consumer"]
        },
        {
            "ticker": "META", "name": "Meta Platforms Inc.",
            "sector": "Communication Services", "industry": "Interactive Media & Services",
            "gics": "50201010", "naics": "519130", "nace": "63.12",
            "products": ["Social Media", "Digital Advertising", "VR/AR"],
            "markets": ["Global", "Social Networking", "Advertising"]
        },
        {
            "ticker": "NFLX", "name": "Netflix, Inc.",
            "sector": "Communication Services", "industry": "Entertainment",
            "gics": "50202010", "naics": "512110", "nace": "59.11",
            "products": ["Streaming Service", "Content Production"],
            "markets": ["Global", "Consumer Entertainment"]
        },
        # Consumer Discretionary
        {
            "ticker": "AMZN", "name": "Amazon.com Inc.",
            "sector": "Consumer Discretionary", "industry": "Broadline Retail",
            "gics": "25502010", "naics": "454110", "nace": "47.91",
            "products": ["E-commerce", "Cloud Computing", "Streaming", "Logistics"],
            "markets": ["Global", "Retail", "Enterprise"]
        },
        {
            "ticker": "BABA", "name": "Alibaba Group Holding Limited",
            "sector": "Consumer Discretionary", "industry": "Broadline Retail",
            "gics": "25502010", "naics": "454110", "nace": "47.91",
            "products": ["E-commerce", "Cloud Computing", "Digital Media"],
            "markets": ["China", "Global", "Retail"]
        },
        {
            "ticker": "TSLA", "name": "Tesla Inc.",
            "sector": "Consumer Discretionary", "industry": "Automobiles",
            "gics": "25102010", "naics": "336111", "nace": "29.10",
            "products": ["Electric Vehicles", "Energy Storage", "Solar Panels"],
            "markets": ["Global", "Automotive", "Energy"]
        },
        {
            "ticker": "F", "name": "Ford Motor Company",
            "sector": "Consumer Discretionary", "industry": "Automobiles",
            "gics": "25102010", "naics": "336111", "nace": "29.10",
            "products": ["Automobiles", "Trucks", "Financial Services"],
            "markets": ["Global", "Automotive"]
        },
        # Semiconductors
        {
            "ticker": "NVDA", "name": "NVIDIA Corporation",
            "sector": "Information Technology", "industry": "Semiconductors & Semiconductor Equipment",
            "gics": "45301020", "naics": "334413", "nace": "26.11",
            "products": ["GPUs", "AI Chips", "Data Center Hardware"],
            "markets": ["Global", "Gaming", "Data Center", "Automotive"]
        },
        {
            "ticker": "AMD", "name": "Advanced Micro Devices, Inc.",
            "sector": "Information Technology", "industry": "Semiconductors & Semiconductor Equipment",
            "gics": "45301020", "naics": "334413", "nace": "26.11",
            "products": ["CPUs", "GPUs", "Semi-custom SoCs"],
            "markets": ["Global", "PC", "Data Center", "Gaming"]
        },
        {
            "ticker": "INTC", "name": "Intel Corporation",
            "sector": "Information Technology", "industry": "Semiconductors & Semiconductor Equipment",
            "gics": "45301020", "naics": "334413", "nace": "26.11",
            "products": ["CPUs", "FPGAs", "Networking Chips"],
            "markets": ["Global", "PC", "Data Center"]
        },
        # Financials
        {
            "ticker": "JPM", "name": "JPMorgan Chase & Co.",
            "sector": "Financials", "industry": "Banks",
            "gics": "40101010", "naics": "522110", "nace": "64.19",
            "products": ["Investment Banking", "Commercial Banking", "Asset Management"],
            "markets": ["Global", "Finance"]
        },
        {
            "ticker": "BAC", "name": "Bank of America Corp",
            "sector": "Financials", "industry": "Banks",
            "gics": "40101010", "naics": "522110", "nace": "64.19",
            "products": ["Banking", "Wealth Management"],
            "markets": ["Global", "Finance"]
        },
        {
            "ticker": "V", "name": "Visa Inc.",
            "sector": "Financials", "industry": "Financial Services",
            "gics": "40201030", "naics": "522320", "nace": "66.19",
            "products": ["Payments", "Transaction Processing"],
            "markets": ["Global", "Payments"]
        },
        {
            "ticker": "MA", "name": "Mastercard Incorporated",
            "sector": "Financials", "industry": "Financial Services",
            "gics": "40201030", "naics": "522320", "nace": "66.19",
            "products": ["Payments", "Transaction Processing"],
            "markets": ["Global", "Payments"]
        },
         # Consumer Staples
        {
            "ticker": "PG", "name": "Procter & Gamble Company",
            "sector": "Consumer Staples", "industry": "Household Products",
            "gics": "30301010", "naics": "325611", "nace": "20.41",
            "products": ["Baby Care", "Fabric Care", "Family Care", "Beauty"],
            "markets": ["Global", "Consumer Goods"]
        },
        {
            "ticker": "CL", "name": "Colgate-Palmolive Company",
            "sector": "Consumer Staples", "industry": "Household Products",
            "gics": "30301010", "naics": "325611", "nace": "20.41",
            "products": ["Oral Care", "Personal Care", "Home Care"],
            "markets": ["Global", "Consumer Goods"]
        },
        {
            "ticker": "KO", "name": "Coca-Cola Company",
            "sector": "Consumer Staples", "industry": "Beverages",
            "gics": "30201010", "naics": "312111", "nace": "11.07",
            "products": ["Soft Drinks", "Water", "Juice"],
            "markets": ["Global", "Beverages"]
        },
        {
            "ticker": "PEP", "name": "PepsiCo Inc.",
            "sector": "Consumer Staples", "industry": "Beverages",
            "gics": "30201010", "naics": "312111", "nace": "11.07",
            "products": ["Soft Drinks", "Snacks", "Food"],
            "markets": ["Global", "Beverages", "Snacks"]
        }
    ]

    def __init__(self, config: Dict[str, Any], **kwargs):
        super().__init__(config, **kwargs)

    async def execute(self, input_data: Union[str, AgentInput, Dict[str, Any]] = None, **kwargs) -> Union[Dict[str, Any], AgentOutput]:
        """
        Executes peer identification based on criteria.
        """
        query = ""
        is_standard_mode = False
        method = "gics"  # Default method
        target_ticker = ""

        if input_data is not None:
            if isinstance(input_data, AgentInput):
                query = input_data.query
                is_standard_mode = True
                if input_data.context:
                    method = input_data.context.get("method", "gics")
                    target_ticker = input_data.context.get("ticker", "")
            elif isinstance(input_data, str):
                query = input_data
            elif isinstance(input_data, dict):
                query = input_data.get("query", "")
                method = input_data.get("method", "gics")
                target_ticker = input_data.get("ticker", "")
                kwargs.update(input_data)

        # Extract ticker logic (same as before)
        if not target_ticker:
            parts = query.split()
            for p in parts:
                clean_p = p.strip().upper().replace(",", "").replace("?", "")
                # Relaxed check for ticker-like pattern if yfinance is available
                if (any(c['ticker'] == clean_p for c in self.COMPANY_DB) or (yf and 2 <= len(clean_p) <= 5 and clean_p.isalpha())):
                    target_ticker = clean_p
                    break

        if not target_ticker:
            clean_q = query.strip().upper()
            if (any(c['ticker'] == clean_q for c in self.COMPANY_DB) or (yf and 2 <= len(clean_q) <= 5 and clean_q.isalpha())):
                target_ticker = clean_q

        if not target_ticker:
            msg = "Could not identify a valid target ticker for peer analysis."
            if is_standard_mode:
                return AgentOutput(answer=msg, sources=[], confidence=0.0, metadata={"error": "Missing Ticker"})
            return {"error": msg}

        logger.info(f"Building peer set for {target_ticker} using method: {method}")

        # Async execution if needed, though fetch_real_data is sync, so using executor in find_peers if we wanted to
        peers = await self.find_peers(target_ticker, method)

        # Build Report
        # Try to find target info in peers or DB
        target_info = next((p for p in peers if p.get('ticker') == target_ticker), None)
        if not target_info:
             # Fallback to DB or fetch
             target_info = next((c for c in self.COMPANY_DB if c['ticker'] == target_ticker), {"name": target_ticker})

        report = f"Peer Set Analysis for {target_ticker} ({target_info.get('name', 'Unknown')})\n"
        report += f"Method: {method.upper()} Classification\n"
        report += f"Sector: {target_info.get('sector', 'N/A')} | Industry: {target_info.get('industry', 'N/A')}\n"
        report += "-" * 40 + "\n"

        if not peers:
            report += "No peers found using this criteria.\n"
        else:
            # Sort by similarity if available
            sorted_peers = sorted(peers, key=lambda x: x.get('similarity', 0), reverse=True)
            for p in sorted_peers:
                if p['ticker'] == target_ticker: continue # Skip self in list

                report += f"- {p['ticker']}: {p.get('name', 'N/A')}"
                if 'similarity' in p:
                     report += f" (Similarity: {p['similarity']:.2f})"
                else:
                     report += f" ({p.get('industry', 'N/A')})"
                report += "\n"

                if method == 'products':
                    if 'products' in p and 'products' in target_info:
                         overlap = set(target_info.get('products', [])) & set(p.get('products', []))
                         if overlap:
                             report += f"  Overlap: {', '.join(overlap)}\n"

        result = {
            "target": target_ticker,
            "method": method,
            "peers": peers,
            "count": len(peers)
        }

        if is_standard_mode:
            return AgentOutput(
                answer=report,
                sources=["yfinance" if yf else "Mock DB", "Internal Classification DB"],
                confidence=0.95 if peers else 0.0,
                metadata=result
            )

        return result

    async def find_peers(self, ticker: str, method: str = "gics") -> List[Dict[str, Any]]:
        """
        Finds peers using real data (if available) or fallback to mock DB.
        """
        loop = asyncio.get_running_loop()

        # 1. Try to fetch real data for target
        target_data = await loop.run_in_executor(None, self.fetch_real_data, ticker)

        # If fetch failed or yf missing, fallback to mock DB
        if not target_data:
            logger.info(f"Real data unavailable for {ticker}, falling back to Mock DB.")
            return self.find_peers_mock(ticker, method)

        # 2. Peer Matching Strategy with Real Data
        peers = []

        # Get candidate pool (Mock DB + potentially extended list)
        candidates = self.COMPANY_DB

        if method == "semantic" or method == "products":
             if not TfidfVectorizer or not target_data.get('summary'):
                 logger.warning("Semantic analysis unavailable (missing sklearn or summary). Falling back to mock products.")
                 return self.find_peers_mock(ticker, 'products')

             # Fetch summaries for candidates (limited to mock DB for performance in this agent)
             # In production, this would query a vector DB.
             documents = [target_data['summary']]
             valid_candidates = []

             # Augment candidates with real data if possible (slow)
             # For this demo, we'll just check against our mock DB's 'sector' to filter first?
             # Or just use the mock DB entries and if we can fetch summary for them.
             # Fetching 20 summaries is slow.
             # OPTIMIZATION: Just use the mock DB names as corpus if summaries unavailable,
             # OR trust the Mock DB's "products" field if real summary fetch is too heavy.

             # Let's try to fetch real data for candidates that share the same sector in Mock DB
             # If target_data has sector, filter Mock DB by that sector
             target_sector = target_data.get('sector')
             filtered_candidates = [c for c in candidates if c.get('sector') == target_sector or c.get('sector') is None]

             # Fetch summaries for these filtered candidates
             # This is still potentially slow (serial fetches).
             # We will just fallback to Mock DB products if semantic requested but we can't vector search.
             # Actually, let's implement a hybrid:
             # If we have summary, we can try to compute similarity against Mock DB 'products' keywords stringified.

             corpus = [target_data['summary']]
             valid_candidates = []

             for c in filtered_candidates:
                 if c['ticker'] == ticker: continue
                 # Synthesize a "summary" from products/industry
                 text = f"{c.get('industry', '')} {' '.join(c.get('products', []))} {c.get('markets', '')}"
                 corpus.append(text)
                 valid_candidates.append(c)

             tfidf = TfidfVectorizer().fit_transform(corpus)
             cosine_similarities = cosine_similarity(tfidf[0:1], tfidf).flatten()

             # skip first (self)
             for i, score in enumerate(cosine_similarities[1:]):
                 candidate = valid_candidates[i]
                 if score > 0.1: # Threshold
                     candidate['similarity'] = float(score)
                     peers.append(candidate)

        else:
            # Code/Sector/Industry matching using Real Data
            target_val = target_data.get(method) if method in target_data else target_data.get('industry') if method in ['gics', 'naics', 'nace'] else None

            if not target_val:
                 # Fallback to mock
                 return self.find_peers_mock(ticker, method)

            # Compare against Mock DB (since we don't have a real DB of all companies)
            # We assume Mock DB has correct codes.
            # Warning: Target might use "Technology Hardware..." but Mock DB might have slight variance if not normalized.
            # We'll do exact string match for sector/industry
            for company in candidates:
                if company['ticker'] == ticker: continue

                # If method is sector/industry, check match
                if method in ['sector', 'industry']:
                    if company.get(method) == target_val:
                        peers.append(company)
                else:
                     # GICS/NAICS/NACE -> Mock DB has these. Target Real Data might NOT have them easily from yfinance info
                     # yf info has 'sector', 'industry', 'industryKey', 'sectorKey'. It usually doesn't have GICS/NAICS codes directly exposed in 'info' reliably.
                     # So if method is classification code, we might have to rely on Mock DB for target too.
                     pass

            # If no peers found via Real Data attributes, fallback to Mock
            if not peers:
                 return self.find_peers_mock(ticker, method)

        return peers

    def find_peers_mock(self, ticker: str, method: str) -> List[Dict[str, Any]]:
        """
        Original logic using only Mock DB.
        """
        target = next((c for c in self.COMPANY_DB if c['ticker'] == ticker), None)
        if not target:
            return []

        peers = []
        if method == 'products' or method == 'semantic':
            target_products = set(target.get('products', []))
            for company in self.COMPANY_DB:
                if company['ticker'] == ticker: continue
                company_products = set(company.get('products', []))
                if not target_products.isdisjoint(company_products):
                    peers.append(company)
        else:
            target_code = target.get(method)
            if not target_code:
                if method in ['gics', 'naics', 'nace']:
                    target_code = target.get('industry')
                    method = 'industry'

            for company in self.COMPANY_DB:
                if company['ticker'] == ticker: continue
                if company.get(method) == target_code:
                    peers.append(company)
        return peers

    def fetch_real_data(self, ticker: str) -> Optional[Dict[str, Any]]:
        """
        Fetches data from yfinance if available.
        """
        if not yf:
            return None
        try:
            t = yf.Ticker(ticker)
            info = t.info
            # Map yfinance info to our schema
            return {
                "ticker": ticker,
                "name": info.get("longName"),
                "sector": info.get("sector"),
                "industry": info.get("industry"),
                "summary": info.get("longBusinessSummary"),
                "products": [], # yfinance doesn't give products list
                "markets": []
            }
        except Exception as e:
            logger.warning(f"yfinance fetch failed for {ticker}: {e}")
            return None
