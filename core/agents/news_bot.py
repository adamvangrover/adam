import sys
import os
import json
import time
import random
import asyncio
import argparse
from typing import Dict, Any, Optional, List, Union, Set
from collections import defaultdict
from datetime import datetime, timezone

# --- Dependency Management ---

# 1. Async HTTP (Required)
try:
    import httpx
except ImportError:
    print("CRITICAL: 'httpx' is required. Please run: pip install httpx")
    sys.exit(1)

# 2. RSS Feeds (Optional)
try:
    import feedparser
except ImportError:
    print("NOTICE: 'feedparser' not found. RSS features disabled.")
    feedparser = None

# 3. Crypto Data (Optional)
try:
    from pycoingecko import CoinGeckoAPI
except ImportError:
    print("NOTICE: 'pycoingecko' not found. Crypto features disabled.")
    CoinGeckoAPI = None

# 4. NLP & Machine Learning (Optional)
import nltk
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM
    TORCH_AVAILABLE = True
except ImportError:
    print("NOTICE: PyTorch/Transformers not found. AI Sentiment/Summarization disabled (using fallbacks).")
    torch = None
    TORCH_AVAILABLE = False

# 5. Core Framework (Optional - for integration context)
try:
    from semantic_kernel import Kernel
    from core.agents.agent_base import AgentBase
except ImportError:
    # Stub for standalone mode
    class AgentBase:
        def __init__(self, config, kernel):
            self.config = config
            self.kernel = kernel
        def get_skill_schema(self): return {}
        async def receive_message(self, sender, message): return message
    Kernel = Any


class NewsBot(AgentBase):
    """
    An advanced News Aggregation Agent that fetches data from APIs, RSS, and Crypto sources,
    performs AI-based sentiment analysis, summarizes content, and filters for user portfolios.
    """

    def __init__(self, config: Dict[str, Any], kernel: Optional[Kernel] = None):
        super().__init__(config, kernel)

        # Configuration Loading
        self.user_preferences = self.config.get('user_preferences', {})
        self.news_api_key = self.config.get('news_api_key', os.getenv("NEWS_API_KEY"))
        self.search_api_key = self.config.get('search_api_key', os.getenv("SEARCH_API_KEY"))
        self.portfolio = self.config.get('portfolio', {})
        self.user_api_sources = self.config.get('user_api_sources', [])
        
        # Runtime State
        self.aggregated_news: List[Dict[str, Any]] = []
        self.seen_alert_urls: Set[str] = set()
        self.custom_news_sources = self._load_custom_sources()

        # Service Initialization
        self.cg = CoinGeckoAPI() if CoinGeckoAPI else None
        
        # Validation
        if not self.news_api_key:
            print("WARNING: 'news_api_key' missing. NewsAPI calls will fail.")
        
        # ML Model Placeholders
        self.finbert_tokenizer = None
        self.finbert_model = None
        self.summarizer_tokenizer = None
        self.summarizer_model = None

        # Initialization
        self._ensure_nltk_resources()
        if TORCH_AVAILABLE:
            self._load_ml_models()

    def _load_custom_sources(self) -> Dict[str, str]:
        """Normalize custom sources into a dict."""
        sources = {}
        for source in self.user_api_sources:
            if isinstance(source, dict) and 'name' in source and 'url' in source:
                sources[source['name']] = source['url']
        return sources

    def _ensure_nltk_resources(self):
        """Ensure minimal NLTK data exists for text splitting."""
        try:
            nltk.data.find('tokenizers/punkt')
        except (LookupError, nltk.downloader.DownloadError):
            print("Downloading NLTK 'punkt' tokenizer...")
            nltk.download('punkt', quiet=True)

    def _load_ml_models(self):
        """Loads FinBERT and BART models if hardware permits."""
        print("Initializing AI Models... (This runs once)")
        
        # Sentiment Model (FinBERT)
        try:
            name = "ProsusAI/finbert"
            # nosec B615: Model is pinned to main revision for now; in production use specific hash
            self.finbert_tokenizer = AutoTokenizer.from_pretrained(name, revision="main")  # nosec B615
            self.finbert_model = AutoModelForSequenceClassification.from_pretrained(name, revision="main")  # nosec B615
            print(f"✔ Sentiment Model ({name}) loaded.")
        except Exception as e:
            print(f"✘ Failed to load FinBERT: {e}")

        # Summarization Model (DistilBART)
        try:
            # Using distilbart-cnn-12-6 for speed/memory efficiency
            name = "sshleifer/distilbart-cnn-12-6"
            # nosec B615: Model is pinned to main revision for now; in production use specific hash
            self.summarizer_tokenizer = AutoTokenizer.from_pretrained(name, revision="main")  # nosec B615
            self.summarizer_model = AutoModelForSeq2SeqLM.from_pretrained(name, revision="main")  # nosec B615
            print(f"✔ Summarization Model ({name}) loaded.")
        except Exception as e:
            print(f"✘ Failed to load Summarizer: {e}")

    # --- AGGREGATION LOGIC ---

    async def aggregate_news(self) -> List[Dict[str, Any]]:
        """Main entry point to fetch news from all configured sources concurrently."""
        tasks = []
        topics = self.user_preferences.get('topics', [])
        print(f"Fetching news for topics: {topics}...")

        # 1. Crypto
        if 'crypto' in topics and self.cg:
            tasks.append(self.get_crypto_news())

        # 2. General Finance (NewsAPI)
        # Map topics to search queries
        topic_map = {
            'finance': 'finance',
            'stocks': 'stocks',
            'commodities': 'commodities OR gold OR oil',
            'treasuries': 'treasury bonds OR yield curve',
            'forex': 'forex OR currency market'
        }
        
        for topic, query in topic_map.items():
            if topic in topics:
                tasks.append(self._fetch_from_newsapi(query))

        # 3. RSS
        if feedparser:
            tasks.append(self.get_reuters_business_news_rss())

        # 4. Custom Sources
        for source_name, source_url in self.custom_news_sources.items():
            tasks.append(self.get_custom_news(source_url))

        # Execute
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        all_news = []
        for res in results:
            if isinstance(res, list):
                all_news.extend(res)
            elif isinstance(res, Exception):
                print(f"Error in data fetch: {res}")

        # Post-process: Filter by portfolio
        self.aggregated_news = self.filter_news_by_portfolio(all_news)
        return self.aggregated_news

    async def _fetch_from_newsapi(self, query: str) -> List[Dict[str, Any]]:
        """Helper to fetch from NewsAPI using httpx (Async)."""
        if not self.news_api_key:
            return []
        
        url = "https://newsapi.org/v2/everything"
        params = {
            'q': query,
            'apiKey': self.news_api_key,
            'language': 'en',
            'sortBy': 'relevancy',
            'pageSize': 15 
        }

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(url, params=params, timeout=10.0)
                if resp.status_code == 429:
                    print(f"NewsAPI Rate Limit Reached for query: {query}")
                    return []
                resp.raise_for_status()
                return resp.json().get('articles', [])
        except Exception as e:
            print(f"NewsAPI Error ({query}): {e}")
            return []

    async def get_crypto_news(self) -> List[Dict[str, Any]]:
        """Fetches trending crypto coins (Runs blocking code in thread)."""
        if not self.cg: return []
        try:
            # Run blocking call in a separate thread
            data = await asyncio.to_thread(self.cg.get_trending_searches)
            news_items = []
            if 'coins' in data:
                for coin in data['coins']:
                    item = coin.get('item', {})
                    news_items.append({
                        'title': f"Trending Crypto: {item.get('name')} ({item.get('symbol')})",
                        'description': f"Rank: {item.get('market_cap_rank')}, Price BTC: {item.get('price_btc')}",
                        'url': f"https://www.coingecko.com/en/coins/{item.get('id')}",
                        'source': {'name': 'CoinGecko Trending'},
                        'published_at': datetime.now(timezone.utc).isoformat()
                    })
            return news_items
        except Exception as e:
            print(f"CoinGecko Error: {e}")
            return []

    async def get_reuters_business_news_rss(self) -> List[Dict[str, Any]]:
        """Parses Reuters RSS Feed."""
        if not feedparser: return []
        url = "http://feeds.reuters.com/reuters/businessNews"
        try:
            feed = await asyncio.to_thread(feedparser.parse, url)
            items = []
            for entry in feed.entries:
                # Normalize time
                pub_date = datetime.now(timezone.utc).isoformat()
                if hasattr(entry, 'published_parsed') and entry.published_parsed:
                    pub_date = datetime.fromtimestamp(time.mktime(entry.published_parsed), tz=timezone.utc).isoformat()

                items.append({
                    'title': entry.get('title', 'No Title'),
                    'description': entry.get('summary', ''),
                    'link': entry.get('link', ''),
                    'source': {'name': 'Reuters RSS'},
                    'published_at': pub_date
                })
            return items
        except Exception as e:
            print(f"RSS Error: {e}")
            return []

    async def get_custom_news(self, url: str) -> List[Dict[str, Any]]:
        """Fetch news from a custom JSON endpoint."""
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(url, timeout=10.0)
                resp.raise_for_status()
                data = resp.json()
                # Attempt to find list in common keys
                return data.get('articles', data.get('news', data.get('data', [])))
        except Exception as e:
            print(f"Custom Source Error ({url}): {e}")
            return []

    # --- PROCESSING & AI ANALYSIS ---

    def filter_news_by_portfolio(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filters articles based on portfolio holdings (Case insensitive)."""
        if not self.portfolio:
            return articles # Return all if no portfolio
        
        # Flatten portfolio
        targets = set()
        if isinstance(self.portfolio, list):
            targets.update(self.portfolio)
        elif isinstance(self.portfolio, dict):
            for v in self.portfolio.values():
                if isinstance(v, list): targets.update(v)
        
        targets = {str(t).lower() for t in targets}
        if not targets: return articles

        filtered = []
        for art in articles:
            text = (art.get('title', '') + " " + (art.get('description') or "")).lower()
            if any(t in text for t in targets):
                filtered.append(art)
        
        # If strict filtering returns nothing, fall back to returning everything? 
        # For now, let's keep strict filtering to reduce noise.
        return filtered

    def analyze_sentiment(self, article: Dict[str, Any]) -> float:
        """
        Returns sentiment score: -1.0 (Negative) to 1.0 (Positive).
        Uses FinBERT if available, otherwise returns 0.0.
        """
        if not self.finbert_model or not TORCH_AVAILABLE:
            return 0.0

        text = (article.get('title', '') + " " + (article.get('description') or ""))[:512]
        if not text.strip(): return 0.0

        try:
            inputs = self.finbert_tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
            with torch.no_grad():
                outputs = self.finbert_model(**inputs)
            
            # ProsusAI/finbert labels: [positive, negative, neutral]
            probs = torch.softmax(outputs.logits, dim=-1)
            pos, neg = probs[0][0].item(), probs[0][1].item()
            neu = probs[0][2].item()

            if pos > neg and pos > neu: return pos
            if neg > pos and neg > neu: return -neg
            return 0.0
        except Exception:
            return 0.0

    def personalize_feed(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Calculates Impact Score and Sorts."""
        processed = []
        for art in articles:
            # 1. Sentiment
            sentiment = self.analyze_sentiment(art)
            art['sentiment_score'] = sentiment

            # 2. Portfolio Relevance
            relevance = 1.0
            targets = [] 
            if isinstance(self.portfolio, dict):
                for v in self.portfolio.values(): targets.extend(v)
            elif isinstance(self.portfolio, list):
                targets = self.portfolio
            
            text = (art.get('title', '') + " " + (art.get('description') or "")).lower()
            for t in targets:
                if str(t).lower() in text:
                    relevance += 1.0 # Boost score if portfolio item mentioned

            # 3. Final Impact Score
            # Magnitude of sentiment * relevance. High negative is just as "impactful" as high positive.
            art['impact_score'] = sentiment * relevance
            processed.append(art)
        
        # Sort by absolute impact (ignoring sign) to show most significant news first?
        # Or sort by highest positive? Let's sort by raw impact (Positive first, Negative last)
        # However, for alerts, we want extremes. 
        # Let's sort by Magnitude for relevance.
        processed.sort(key=lambda x: abs(x.get('impact_score', 0)), reverse=True)
        return processed

    async def summarize_articles(self, articles: List[Dict[str, Any]]) -> str:
        """Summarizes top articles using BART or fallback."""
        if not articles: return "No articles to summarize."

        # Prepare text
        text_content = " ".join([
            (a.get('description') or a.get('title', '')) for a in articles[:5]
        ])

        # AI Summary
        if self.summarizer_model and TORCH_AVAILABLE:
            try:
                inputs = self.summarizer_tokenizer(text_content, return_tensors="pt", max_length=1024, truncation=True)
                summary_ids = self.summarizer_model.generate(
                    inputs.input_ids, num_beams=4, max_length=150, early_stopping=True
                )
                return self.summarizer_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            except Exception as e:
                print(f"Summarizer failed: {e}. Using fallback.")

        # Fallback Summary
        try:
            sentences = nltk.sent_tokenize(text_content)
            return " ".join(sentences[:3])
        except:
            return text_content[:300] + "..."

    # --- REPORTING & ALERTS ---

    def send_alerts(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prints alerts for high-impact news."""
        thresholds = self.config.get('alerting_thresholds', {})
        pos_thresh = thresholds.get('positive_impact', 0.5)
        neg_thresh = thresholds.get('negative_impact', -0.5)

        alerted_articles = []
        for art in articles:
            score = art.get('impact_score', 0)
            url = art.get('link', art.get('url', art.get('title')))
            
            if url in self.seen_alert_urls: continue

            label = ""
            if score > pos_thresh: label = "🚀 OPPORTUNITY"
            elif score < neg_thresh: label = "⚠️ RISK ALERT"

            if label:
                print(f"\n[{label}] {art.get('title')}\nScore: {score:.2f} | {url}")
                self.seen_alert_urls.add(url)
                alerted_articles.append(art)
        
        return alerted_articles

    def generate_report(self, articles: List[Dict[str, Any]], summary: str) -> str:
        """Compiles a text report."""
        lines = [
            "--------------------------------------------------",
            f" NEWS INTELLIGENCE REPORT | {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
            "--------------------------------------------------",
            "\n[EXECUTIVE SUMMARY]",
            summary,
            "\n[TOP STORIES]"
        ]
        
        for i, art in enumerate(articles[:5]):
            lines.append(f"{i+1}. {art.get('title')}")
            lines.append(f"   Source: {art.get('source', {}).get('name', 'Unknown')}")
            lines.append(f"   Impact: {art.get('impact_score', 0):.2f}")
            lines.append(f"   Link: {art.get('link', art.get('url', ''))}")
        
        lines.append("--------------------------------------------------")
        return "\n".join(lines)

    # --- EXECUTION ---

    async def execute(self, *args, **kwargs) -> Dict[str, Any]:
        """Run a full cycle: Fetch -> Analyze -> Alert -> Report."""
        print(f"\n--- Cycle Start: {datetime.now().strftime('%H:%M:%S')} ---")
        
        news = await self.aggregate_news()
        personalized = self.personalize_feed(news)
        self.send_alerts(personalized)
        
        report = None
        if self.config.get('enable_analysis_reporting', True) and personalized:
            summary = await self.summarize_articles(personalized)
            report = self.generate_report(personalized, summary)
            
        return {
            "feed": personalized,
            "report": report
        }

    async def monitor(self, duration_minutes: int = 5, interval_seconds: int = 60):
        """Runs the bot in a loop."""
        end_time = time.time() + (duration_minutes * 60)
        print(f"Monitoring started for {duration_minutes}m (Interval: {interval_seconds}s)...")
        
        while time.time() < end_time:
            await self.execute()
            
            remaining = end_time - time.time()
            if remaining <= 0: break
            
            sleep_time = min(remaining, interval_seconds)
            print(f"Sleeping for {int(sleep_time)}s...")
            await asyncio.sleep(sleep_time)


# --- STANDALONE RUNNER ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NewsBot Standalone")
    parser.add_argument("--key", help="NewsAPI Key", required=False)
    parser.add_argument("--portfolio", help="JSON string or path to file", default="{}")
    parser.add_argument("--monitor", action="store_true", help="Run in continuous monitor mode")
    parser.add_argument("--report", action="store_true", help="Print analysis report")
    args = parser.parse_args()

    # Load Portfolio
    portfolio = {}
    if args.portfolio:
        try:
            if os.path.exists(args.portfolio):
                with open(args.portfolio, 'r') as f: portfolio = json.load(f)
            else:
                portfolio = json.loads(args.portfolio)
        except Exception:
            print("Error parsing portfolio JSON. Using empty portfolio.")

    # Config
    config = {
        "news_api_key": args.key or os.getenv("NEWS_API_KEY"),
        "user_preferences": {"topics": ["finance", "crypto", "stocks"]},
        "portfolio": portfolio,
        "enable_analysis_reporting": args.report,
        "alerting_thresholds": {"positive_impact": 0.5, "negative_impact": -0.5}
    }

    if not config["news_api_key"]:
        print("Error: No NewsAPI key provided via --key or env var NEWS_API_KEY.")
        sys.exit(1)

    bot = NewsBot(config)

    try:
        if args.monitor:
            asyncio.run(bot.monitor())
        else:
            res = asyncio.run(bot.execute())
            if args.report and res['report']:
                print(res['report'])
            elif not args.report:
                print(f"Found {len(res['feed'])} articles.")
                for a in res['feed'][:3]:
                    print(f"- {a['title']} (Imp: {a.get('impact_score',0):.2f})")
    except KeyboardInterrupt:
        print("\nStopped by user.")