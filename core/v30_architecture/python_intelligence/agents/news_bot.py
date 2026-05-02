import logging
import re
import asyncio
import os
import hashlib
import time
import requests
import defusedxml.ElementTree as ET
import threading
from typing import List, Dict, Any, Tuple, Optional
from pydantic import BaseModel

# Mock / graceful fallback environments
MOCK_MODE = os.environ.get("MOCK_MODE", "false").lower() in ("true", "1", "yes")

try:
    import duckdb
    HAS_DUCKDB = True
except ImportError:
    HAS_DUCKDB = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    from duckduckgo_search import DDGS
    HAS_DDGS = True
except ImportError:
    HAS_DDGS = False

try:
    from core.v30_architecture.python_intelligence.agents.base_agent import BaseAgent
except ImportError:
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../')))
    from core.v30_architecture.python_intelligence.agents.base_agent import BaseAgent

logger = logging.getLogger("NewsBot")

class Provenance(BaseModel):
    source_uri: str
    content_hash: str
    timestamp: float

class Telemetry(BaseModel):
    execution_time_ms: float
    chunks_processed: int
    chunks_retrieved: int
    mock_mode_active: bool
    db_active: bool
    nlp_active: bool
    live_feeds_active: bool

class NewsChunk(BaseModel):
    chunk_id: str
    symbol: str
    source: str
    doc_type: str
    content: str
    original_title: str
    provenance: Provenance
    macro_themes: List[str]
    relevance_score: float

class ActionableInsight(BaseModel):
    symbol: str
    sentiment: float
    conviction: float
    insight: str
    synthesized_summary: str
    market_impact_estimate: str
    macro_themes: List[str]
    supporting_chunks: List[str]
    telemetry: Telemetry

class NewsBotAgent(BaseAgent):
    """
    NewsBotAgent: A highly advanced V30 intelligence agent utilizing a RAG pipeline.

    Capabilities:
    - Live Ingestion (RSS, Web Search) with graceful mock fallback.
    - Semantic chunking with cryptographic provenance tracking.
    - Vectorization (TF-IDF) & In-memory indexing (DuckDB).
    - Multi-layer analysis: filtering, macro-theme tagging, sorting by relevance,
      correlation to financial markets, and conviction scoring.
    """

    def __init__(self, name: str = "NewsBot-V30", role: str = "news_sentiment"):
        super().__init__(name, role)
        self.watchlist: set[str] = {"AAPL", "TSLA", "MSFT"}
        self._db_lock = threading.Lock() # For thread-safety during async executes

        self._symbol_map = {
            re.compile(r'\bapple\b', re.IGNORECASE): "AAPL",
            re.compile(r'\btesla\b', re.IGNORECASE): "TSLA",
            re.compile(r'\bmicrosoft\b', re.IGNORECASE): "MSFT",
        }

        # Macro themes for correlation analysis
        self._macro_themes = {
            "inflation": re.compile(r'\binflation\b|\bcpi\b|\bprices rising\b', re.IGNORECASE),
            "supply_chain": re.compile(r'\bsupply chain\b|\bshortage\b|\bdelay\b', re.IGNORECASE),
            "interest_rates": re.compile(r'\brates\b|\bfed\b|\bhike\b', re.IGNORECASE),
            "innovation": re.compile(r'\bbreakthrough\b|\brelease\b|\bnew product\b', re.IGNORECASE),
            "legal_risk": re.compile(r'\blawsuit\b|\binvestigation\b|\bsec\b|\bregulatory\b', re.IGNORECASE)
        }

        # NLP Setup & Graceful Degradation
        self.is_mock = MOCK_MODE
        self.has_nlp = HAS_SKLEARN and not self.is_mock
        self.live_feeds_active = not self.is_mock

        if self.has_nlp:
            self.vectorizer = TfidfVectorizer(stop_words='english', max_features=100)
        else:
            self.vectorizer = None
            logger.info(f"{self.name}: Running without advanced NLP vectorization.")

        if HAS_DUCKDB and not self.is_mock:
            self.db = duckdb.connect(':memory:')
            self._init_db()
        else:
            self.db = None
            self._mock_db = []
            logger.info(f"{self.name}: Running with mocked DB.")

    def _init_db(self):
        self.db.execute('''
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id VARCHAR PRIMARY KEY,
                symbol VARCHAR,
                source VARCHAR,
                doc_type VARCHAR,
                content VARCHAR,
                original_title VARCHAR,
                source_uri VARCHAR,
                content_hash VARCHAR,
                timestamp DOUBLE,
                macro_themes VARCHAR[],
                relevance_score DOUBLE
            )
        ''')
        self.db.execute('''
            CREATE TABLE IF NOT EXISTS chunk_embeddings (
                chunk_id VARCHAR PRIMARY KEY,
                vector FLOAT[]
            )
        ''')

    def _hash_content(self, content: str) -> str:
        return hashlib.sha256(content.encode('utf-8')).hexdigest()

    async def execute(self, **kwargs) -> List[Dict[str, Any]]:
        logger.info(f"{self.name} executing multi-layer advanced RAG cycle...")
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.run_cycle)

    def run_cycle(self) -> List[Dict[str, Any]]:
        start_time = time.time()

        # 1. Orchestrated Ingestion
        raw_feed = self._ingest_data()

        # 2. Semantic Chunking, Tagging, and Relevance Scoring
        all_chunks = []
        for doc in raw_feed:
            chunks = self._process_document(doc)
            all_chunks.extend(chunks)

        if not all_chunks:
            return []

        # 3. Embedding & Vectorization
        embeddings = []
        if self.has_nlp:
            corpus = [c.content for c in all_chunks]
            tfidf_matrix = self.vectorizer.fit_transform(corpus).toarray()
            embeddings = tfidf_matrix.tolist()

        # 4. Index & Map (thread-safe)
        with self._db_lock:
            self._index_chunks(all_chunks, embeddings)

        # 5. Multi-Layer Synthesis & Insight Generation
        insights = []
        for symbol in self.watchlist:
            # Retrieve chunks specifically for this symbol (thread-safe)
            with self._db_lock:
                retrieved_chunks = self._retrieve_chunk_objects(symbol)

            if not retrieved_chunks:
                continue

            # Analyze, synthesize, and determine conviction
            insight_obj = self._analyze_and_synthesize(symbol, retrieved_chunks)
            if not insight_obj:
                continue

            # Attach Telemetry
            exec_time = (time.time() - start_time) * 1000
            insight_obj.telemetry = Telemetry(
                execution_time_ms=exec_time,
                chunks_processed=len(all_chunks),
                chunks_retrieved=len(retrieved_chunks),
                mock_mode_active=self.is_mock,
                db_active=bool(self.db),
                nlp_active=self.has_nlp,
                live_feeds_active=self.live_feeds_active
            )
            insights.append(insight_obj)

        return [i.model_dump() for i in insights]

    def _process_document(self, document: Dict[str, str]) -> List[NewsChunk]:
        """Parses document into semantic chunks, tags macro themes, and assigns relevance."""
        title = document.get("title", "")
        content = document.get("content", "")
        source = document.get("source", "unknown")
        doc_type = document.get("doc_type", "article")

        full_text = f"{title}. {content}"
        found_symbols = set()
        for pattern, symbol in self._symbol_map.items():
            if pattern.search(full_text):
                found_symbols.add(symbol)

        chunks = []
        sentences = [s.strip() for s in content.split('.') if s.strip()]

        for sym in found_symbols:
            for idx, sentence in enumerate(sentences):
                if not sentence: continue

                # Identify Macro Themes
                themes = []
                for theme_name, regex in self._macro_themes.items():
                    if regex.search(sentence):
                        themes.append(theme_name)

                # Base relevance on length, explicit symbol mention, and macro themes
                relevance = 0.1 # base
                if sym.lower() in sentence.lower():
                    relevance += 0.4
                if themes:
                    relevance += 0.3
                if len(sentence) > 30:
                    relevance += 0.2

                content_hash = self._hash_content(sentence)
                chunk_id = f"{sym}_{source}_{idx}_{content_hash[:8]}"

                prov = Provenance(
                    source_uri=f"news://{source}/{sym}/{idx}",
                    content_hash=content_hash,
                    timestamp=time.time()
                )

                chunks.append(NewsChunk(
                    chunk_id=chunk_id, symbol=sym, source=source, doc_type=doc_type,
                    content=sentence, original_title=title, provenance=prov,
                    macro_themes=themes, relevance_score=round(relevance, 2)
                ))
        return chunks

    def _analyze_and_synthesize(self, symbol: str, chunks: List[NewsChunk]) -> Optional[ActionableInsight]:
        """
        Multi-layer synthesis: Filters, sorts by relevance, correlates macro themes,
        and calculates weighted sentiment, conviction, and market impact.
        """
        # Filter noise: Must have minimum relevance
        high_rel_chunks = [c for c in chunks if c.relevance_score > 0.2]
        if not high_rel_chunks:
            return None

        # Sort by relevance descending
        sorted_chunks = sorted(high_rel_chunks, key=lambda x: x.relevance_score, reverse=True)

        # Aggregate themes & texts
        all_themes = set()
        combined_text = []
        total_weight = 0.0
        weighted_sentiment = 0.0

        for chunk in sorted_chunks:
            all_themes.update(chunk.macro_themes)
            combined_text.append(chunk.content)

            # Score individual chunk sentiment and weight by relevance
            s_score, _ = self._mock_nlp_scoring(chunk.content)
            weighted_sentiment += (s_score * chunk.relevance_score)
            total_weight += chunk.relevance_score

        # Normalize sentiment [-1.0, 1.0]
        final_sentiment = round(weighted_sentiment / total_weight, 2) if total_weight > 0 else 0.0

        # Calculate Conviction (0.0 to 1.0) based on volume of high relevance chunks & theme density
        base_conviction = min(1.0, len(sorted_chunks) * 0.15)
        theme_bonus = min(0.3, len(all_themes) * 0.1)
        sentiment_strength = abs(final_sentiment) * 0.2
        final_conviction = min(1.0, round(base_conviction + theme_bonus + sentiment_strength, 2))

        # Filter out low conviction noise completely
        if final_conviction < 0.3 and abs(final_sentiment) < 0.3:
            return None

        # Synthesize Summary and Estimate Market Impact
        theme_str = ", ".join(all_themes) if all_themes else "general sentiment"

        if final_sentiment > 0.4:
            impact = f"Probable bullish momentum. Upward price pressure anticipated driven by {theme_str}."
            insight = f"Positive signal detected for {symbol}."
        elif final_sentiment < -0.4:
            impact = f"Probable bearish momentum. Downward price pressure anticipated driven by {theme_str}."
            insight = f"Critical risk alert detected for {symbol}."
        else:
            impact = f"Neutral market impact. Volatility possible based on {theme_str}."
            insight = f"Mixed signals detected for {symbol}."

        summary = f"Synthesized {len(sorted_chunks)} highly relevant chunks revealing {impact}"

        # We attach a placeholder Telemetry; the caller will inject the real one
        placeholder_telemetry = Telemetry(
            execution_time_ms=0.0, chunks_processed=0, chunks_retrieved=0,
            mock_mode_active=False, db_active=False, nlp_active=False, live_feeds_active=False
        )

        return ActionableInsight(
            symbol=symbol,
            sentiment=final_sentiment,
            conviction=final_conviction,
            insight=insight,
            synthesized_summary=summary,
            market_impact_estimate=impact,
            macro_themes=list(all_themes),
            supporting_chunks=[c.chunk_id for c in sorted_chunks],
            telemetry=placeholder_telemetry
        )

    def _index_chunks(self, chunks: List[NewsChunk], embeddings: List[List[float]]):
        if self.db:
            for idx, chunk in enumerate(chunks):
                self.db.execute('''
                    INSERT OR REPLACE INTO chunks
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    chunk.chunk_id, chunk.symbol, chunk.source,
                    chunk.doc_type, chunk.content, chunk.original_title,
                    chunk.provenance.source_uri,
                    chunk.provenance.content_hash, chunk.provenance.timestamp,
                    chunk.macro_themes, chunk.relevance_score
                ))
                if embeddings:
                    self.db.execute('''
                        INSERT OR REPLACE INTO chunk_embeddings
                        VALUES (?, ?)
                    ''', (chunk.chunk_id, embeddings[idx]))
        else:
            self._mock_db.extend(chunks)

    def _retrieve_chunk_objects(self, symbol: str) -> List[NewsChunk]:
        """Retrieves and rehydrates NewsChunk models from DB or mock memory."""
        if self.db:
            res = self.db.execute('SELECT * FROM chunks WHERE symbol = ?', (symbol,)).fetchall()
            chunks = []
            for r in res:
                # Schema: chunk_id, symbol, source, doc_type, content, original_title, source_uri, content_hash, timestamp, macro_themes, relevance_score
                prov = Provenance(source_uri=r[6], content_hash=r[7], timestamp=r[8])
                chunks.append(NewsChunk(
                    chunk_id=r[0], symbol=r[1], source=r[2], doc_type=r[3],
                    content=r[4], original_title=r[5], provenance=prov,
                    macro_themes=r[9] if r[9] else [], relevance_score=r[10]
                ))
            return chunks
        else:
            return [c for c in self._mock_db if c.symbol == symbol]

    def _ingest_data(self) -> List[Dict[str, str]]:
        data = []
        if self.live_feeds_active:
            try:
                rss_data = self._fetch_rss_feeds()
                data.extend(rss_data)
                if HAS_DDGS:
                    web_data = self._fetch_web_search()
                    data.extend(web_data)
            except Exception as e:
                logger.error(f"Live feeds failed: {e}. Degrading to mock data.")
                self.live_feeds_active = False

        if not data:
            data = [
                {"title": "Apple releases revolutionary AR glasses", "content": "The new product release is expected to boost revenue significantly.", "source": "TechCrunch", "doc_type": "news"},
                {"title": "Tesla faces severe supply chain delays", "content": "Production halted due to massive shortage of parts.", "source": "Reuters", "doc_type": "report"},
                {"title": "Microsoft hit with massive SEC lawsuit", "content": "Legal investigation causes stock to drop.", "source": "Wired", "doc_type": "news"}
            ]
        return data

    def _fetch_rss_feeds(self) -> List[Dict[str, str]]:
        data = []
        rss_url = "https://feeds.finance.yahoo.com/rss/2.0/headline?s=AAPL,TSLA,MSFT&region=US&lang=en-US"
        try:
            response = requests.get(rss_url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=5)
            response.raise_for_status()
            xml_content = response.content
            root = ET.fromstring(xml_content)
            for item in root.findall('./channel/item')[:5]:
                title = item.find('title')
                desc = item.find('description')
                data.append({
                    "title": title.text if title is not None else "No Title",
                    "content": desc.text if desc is not None else "",
                    "source": "YahooFinanceRSS",
                    "doc_type": "rss_feed"
                })
        except Exception as e:
            logger.warning(f"Failed to fetch RSS feeds: {e}")
            raise e
        return data

    def _fetch_web_search(self) -> List[Dict[str, str]]:
        data = []
        if not HAS_DDGS:
            return data
        try:
            with DDGS() as ddgs:
                for symbol in self.watchlist:
                    query = f"{symbol} stock news"
                    results = ddgs.text(query, max_results=2)
                    for r in results:
                        data.append({
                            "title": r.get('title', ''),
                            "content": r.get('body', ''),
                            "source": "DDGS_Search",
                            "doc_type": "web_scrape"
                        })
        except Exception as e:
            logger.warning(f"Failed DDGS search: {e}")
            raise e
        return data

    def _mock_nlp_scoring(self, text: str) -> Tuple[float, float]:
        text_lower = text.lower()
        if "delay" in text_lower or "shortage" in text_lower or "lawsuit" in text_lower or "down" in text_lower or "drop" in text_lower or "investigation" in text_lower:
            return -0.85, 0.92
        if "release" in text_lower or "breakthrough" in text_lower or "positive" in text_lower or "up" in text_lower or "rise" in text_lower:
            return 0.80, 0.88
        return 0.0, 0.50
