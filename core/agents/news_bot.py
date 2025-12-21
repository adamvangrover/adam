import requests
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from collections import defaultdict
import time
from pycoingecko import CoinGeckoAPI
import json
import numpy as np
from sklearn.cluster import KMeans
from datetime import datetime
import random
from sklearn.metrics import pairwise_distances_argmin_min

from typing import Dict, Any, Optional, List
import feedparser  # Added import
import time  # For parsing published_at if needed
from datetime import timezone  # For timezone aware datetime objects

from semantic_kernel import Kernel

from core.agents.agent_base import AgentBase
# For get_api_key if needed later, but not for this step
# from core.utils.secrets_utils import get_api_key

import torch  # Added import
from transformers import AutoTokenizer, AutoModelForSequenceClassification  # Added import

# Initialize NLTK sentiment analyzer - REMOVED
# try:
#     nltk.data.find('sentiment/vader_lexicon.zip')
# except nltk.downloader.DownloadError:
#     nltk.download('vader_lexicon')
# sia = SentimentIntensityAnalyzer() - REMOVED

import nltk  # Added import for sentence tokenization in summarizer fallback
from transformers import AutoModelForSeq2SeqLM  # Added import for summarization model

# Define the "NewsBot" class, now inheriting from AgentBase


class NewsBot(AgentBase):
    def __init__(self, config: Dict[str, Any], kernel: Optional[Kernel] = None):
        """
        Initializes the NewsBot Agent.

        Args:
            config (Dict[str, Any]): Agent configuration.
            kernel (Optional[Kernel]): Semantic Kernel instance.
        """
        super().__init__(config, kernel)

        # Extract configurations from self.config (set by AgentBase)
        self.user_preferences = self.config.get('user_preferences', {})
        self.news_api_key = self.config.get('news_api_key', None)
        self.search_api_key = self.config.get('search_api_key', None)  # Added search_api_key
        self.portfolio = self.config.get('portfolio', {})
        self.user_api_sources = self.config.get('user_api_sources', [])

        if not self.news_api_key:
            print("Warning: news_api_key is not configured for NewsBot.")
        if not self.search_api_key:
            print("Warning: search_api_key is not configured for NewsBot. Web search will be unavailable.")

        self.cg = CoinGeckoAPI()
        self.aggregated_news: List[Dict[str, Any]] = []
        self.user_interactions: Dict[str, int] = defaultdict(int)
        self.custom_news_sources = self.load_custom_sources()

        # Load FinBERT model and tokenizer
        self.finbert_tokenizer = None
        self.finbert_model = None
        self.summarizer_tokenizer = None
        self.summarizer_model = None
        self.seen_alert_urls = set()  # For tracking alerted articles in the current session

        # Load FinBERT model and tokenizer
        try:
            model_name_finbert = "ProsusAI/finbert"
            self.finbert_tokenizer = AutoTokenizer.from_pretrained(model_name_finbert)
            self.finbert_model = AutoModelForSequenceClassification.from_pretrained(model_name_finbert)
            print(f"FinBERT model ({model_name_finbert}) loaded successfully.")
        except Exception as e:
            print(f"Error loading FinBERT model: {e}. Sentiment analysis will be impacted.")

        # Load Summarization model and tokenizer
        try:
            model_name_summarizer = "sshleifer/distilbart-cnn-12-6"  # Using a smaller model due to space constraints
            self.summarizer_tokenizer = AutoTokenizer.from_pretrained(model_name_summarizer)
            self.summarizer_model = AutoModelForSeq2SeqLM.from_pretrained(model_name_summarizer)
            print(f"Summarization model ({model_name_summarizer}) loaded successfully.")
        except Exception as e:
            print(f"Error loading Summarization model: {e}. Summarization will use fallback.")
            # Ensure NLTK's punkt is available for fallback
            try:
                nltk.data.find('tokenizers/punkt')
            except nltk.downloader.DownloadError:
                print("NLTK 'punkt' not found. Downloading...")
                nltk.download('punkt', quiet=True)  # quiet=True to avoid verbose output if already there or successful

    def load_custom_sources(self) -> Dict[str, str]:
        """Load custom news APIs provided by the user."""
        custom_sources = {}
        for source in self.user_api_sources:
            custom_sources[source['name']] = source['url']
        return custom_sources

    def aggregate_news(self) -> List[Dict[str, Any]]:  # Added return type hint
        """
        Aggregates news articles from various sources (including custom user sources).

        Returns:
            list: Aggregated news articles
        """
        all_news: List[Dict[str, Any]] = []  # Added type hint

        # Collect crypto-related news
        if 'crypto' in self.user_preferences.get('topics', []):  # Safe get
            all_news.extend(self.get_crypto_news())

        # Collect financial market-related news
        if 'finance' in self.user_preferences.get('topics', []):  # Safe get
            all_news.extend(self.get_finance_news())

        # Collect stock market news
        if 'stocks' in self.user_preferences.get('topics', []):  # Safe get
            all_news.extend(self.get_stock_news())

        # Collect commodities and treasury news
        if 'commodities' in self.user_preferences.get('topics', []):  # Safe get
            all_news.extend(self.get_commodities_news())

        if 'treasuries' in self.user_preferences.get('topics', []):  # Safe get
            all_news.extend(self.get_treasuries_news())

        # Collect FX news
        if 'forex' in self.user_preferences.get('topics', []):  # Safe get
            all_news.extend(self.get_forex_news())

        # Add Reuters Business News RSS
        all_news.extend(self.get_reuters_business_news_rss())

        # Integrate custom sources (user-defined)
        for source_name, source_url in self.custom_news_sources.items():
            all_news.extend(self.get_custom_news(source_url))

        # Filter news based on user's portfolio
        filtered_news = self.filter_news_by_portfolio(all_news)

        return filtered_news

    def get_crypto_news(self) -> List[Dict[str, Any]]:  # Added return type hint
        """Fetch cryptocurrency news using the CoinGeckoAPI."""
        try:
            # Assuming get_trending_searches returns a list of dicts or similar
            # The original pycoingecko might return a more complex structure.
            # This is a simplification.
            trending_searches = self.cg.get_trending_searches()  # Removed language='en' as it's not a standard param
            # Convert trending searches to a news-like format if necessary
            news_items = []
            if isinstance(trending_searches, dict) and 'coins' in trending_searches:
                for coin_info in trending_searches['coins']:
                    item = coin_info.get('item', {})
                    news_items.append({
                        'title': f"Trending: {item.get('name', 'Unknown Coin')} ({item.get('symbol', '')})",
                        'description': f"Market Cap Rank: {item.get('market_cap_rank', 'N/A')}, Score: {item.get('score', 'N/A')}",
                        'source': {'name': 'CoinGecko Trending'},
                        'url': f"https://www.coingecko.com/en/coins/{item.get('id', '')}" if item.get('id') else None
                    })
            return news_items
        except Exception as e:
            print(f"Error fetching crypto news from CoinGecko: {e}")
            return []

    def get_finance_news(self) -> List[Dict[str, Any]]:  # Added return type hint
        """Fetch general finance news using NewsAPI."""
        if not self.news_api_key:
            return []
        url = "https://newsapi.org/v2/everything"
        params = {
            'q': 'finance',
            'apiKey': self.news_api_key,
            'language': 'en',
            'sortBy': 'relevancy'
        }
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()  # Raise an exception for HTTP errors
            return response.json().get('articles', [])
        except requests.exceptions.RequestException as e:
            print(f"Error fetching finance news from NewsAPI: {e}")
            return []

    def get_stock_news(self) -> List[Dict[str, Any]]:  # Added return type hint
        """Fetch stock-related news using an API or data source."""
        if not self.news_api_key:
            return []
        url = "https://newsapi.org/v2/everything"
        params = {
            'q': 'stocks',
            'apiKey': self.news_api_key,
            'language': 'en',
            'sortBy': 'relevancy'
        }
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            return response.json().get('articles', [])
        except requests.exceptions.RequestException as e:
            print(f"Error fetching stock news from NewsAPI: {e}")
            return []

    def get_commodities_news(self) -> List[Dict[str, Any]]:  # Added return type hint
        """Fetch commodities-related news (gold, oil, etc.)."""
        if not self.news_api_key:
            return []
        url = "https://newsapi.org/v2/everything"
        params = {
            'q': 'commodities',
            'apiKey': self.news_api_key,
            'language': 'en',
            'sortBy': 'relevancy'
        }
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            return response.json().get('articles', [])
        except requests.exceptions.RequestException as e:
            print(f"Error fetching commodities news from NewsAPI: {e}")
            return []

    def get_treasuries_news(self) -> List[Dict[str, Any]]:  # Added return type hint
        """Fetch treasury bond-related news."""
        if not self.news_api_key:
            return []
        url = "https://newsapi.org/v2/everything"
        params = {
            'q': 'treasury bonds',
            'apiKey': self.news_api_key,
            'language': 'en',
            'sortBy': 'relevancy'
        }
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            return response.json().get('articles', [])
        except requests.exceptions.RequestException as e:
            print(f"Error fetching treasuries news from NewsAPI: {e}")
            return []

    def get_forex_news(self) -> List[Dict[str, Any]]:  # Added return type hint
        """Fetch foreign exchange news."""
        if not self.news_api_key:
            return []
        url = "https://newsapi.org/v2/everything"
        params = {
            'q': 'forex',
            'apiKey': self.news_api_key,
            'language': 'en',
            'sortBy': 'relevancy'
        }
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            return response.json().get('articles', [])
        except requests.exceptions.RequestException as e:
            print(f"Error fetching forex news from NewsAPI: {e}")
            return []

    def get_custom_news(self, api_url: str) -> List[Dict[str, Any]]:  # Added type hints
        """Fetch custom news from user-provided sources."""
        try:
            response = requests.get(api_url)
            response.raise_for_status()
            return response.json().get('articles', [])
        except requests.exceptions.RequestException as e:
            print(f"Error fetching custom news from {api_url}: {e}")
            return []
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from {api_url}: {e}")
            return []

    def get_reuters_business_news_rss(self) -> List[Dict[str, Any]]:
        """Fetches and parses Reuters Business News RSS feed."""
        feed_url = "http://feeds.reuters.com/reuters/businessNews"
        news_items: List[Dict[str, Any]] = []
        try:
            feed = feedparser.parse(feed_url)
            for entry in feed.entries:
                # Ensure published_parsed is available and convert to ISO format
                published_at = None
                if hasattr(entry, 'published_parsed') and entry.published_parsed:
                    # Create a datetime object, assuming UTC if no timezone info
                    dt_object = datetime.fromtimestamp(time.mktime(entry.published_parsed), tz=timezone.utc)
                    published_at = dt_object.isoformat()
                elif hasattr(entry, 'updated_parsed') and entry.updated_parsed:  # Fallback to updated
                    dt_object = datetime.fromtimestamp(time.mktime(entry.updated_parsed), tz=timezone.utc)
                    published_at = dt_object.isoformat()

                news_items.append({
                    'title': entry.title if hasattr(entry, 'title') else 'No Title',
                    'description': entry.summary if hasattr(entry, 'summary') else 'No Description',
                    'link': entry.link if hasattr(entry, 'link') else '',
                    'published_at': published_at,
                    'source': {'name': 'Reuters Business News'}
                })
        except Exception as e:
            print(f"Error fetching or parsing Reuters RSS feed: {e}")
            # Optionally, log the error more formally or raise it depending on desired handling
        return news_items

    def filter_news_by_portfolio(self, news_articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:  # Added type hints
        """
        Filters news articles based on portfolio holdings (stocks, crypto, etc.).

        Args:
            news_articles (list): List of news articles.

        Returns:
            list: Filtered list of news articles.
        """
        filtered_news: List[Dict[str, Any]] = []  # Added type hint
        for article in news_articles:
            title = article.get('title', '')
            description = article.get('description', '')
            if not title and not description:  # Skip if no content to check
                continue
            for symbol in self.portfolio:
                if symbol.lower() in title.lower() or symbol.lower() in description.lower():
                    filtered_news.append(article)
                    break
        return filtered_news

    def analyze_sentiment(self, article: Dict[str, Any]) -> float:
        """Analyze the sentiment of a news article using FinBERT."""
        text = article.get('title', '') + " " + article.get('description', '')
        if not text.strip():
            return 0.0

        if not self.finbert_tokenizer or not self.finbert_model:
            print("FinBERT model not available. Skipping sentiment analysis.")
            return 0.0  # Neutral score if model isn't loaded

        try:
            inputs = self.finbert_tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)

            with torch.no_grad():  # Disable gradient calculations for inference
                outputs = self.finbert_model(**inputs)

            probs = torch.softmax(outputs.logits, dim=-1)

            # FinBERT typically has labels: 0 for positive, 1 for negative, 2 for neutral
            # Or it might be ['positive', 'negative', 'neutral'] depending on model config.
            # Let's assume the order is positive, negative, neutral from common FinBERT usage.
            # Verify this with model.config.id2label if available and necessary.
            # For "ProsusAI/finbert", the labels are ['positive', 'negative', 'neutral']
            # So, probs[0][0] is positive, probs[0][1] is negative, probs[0][2] is neutral.

            positive_prob = probs[0][0].item()
            negative_prob = probs[0][1].item()
            neutral_prob = probs[0][2].item()

            if positive_prob > negative_prob and positive_prob > neutral_prob:
                return 1.0  # Positive
            elif negative_prob > positive_prob and negative_prob > neutral_prob:
                return -1.0  # Negative
            else:
                return 0.0  # Neutral

        except Exception as e:
            print(f"Error during FinBERT sentiment analysis: {e}")
            return 0.0  # Return neutral on error

    def analyze_impact(self, article: Dict[str, Any]) -> float:
        """
        Calculates an 'impact score' based on sentiment, portfolio relevance, and topic.

        Args:
            article (dict): The news article to evaluate.

        Returns:
            float: The impact score of the article.
        """
        sentiment_score = self.analyze_sentiment(article)
        portfolio_relevance = 0
        title = article.get('title', '')
        description = article.get('description', '')

        # Increase impact score if it matches portfolio holdings
        for symbol in self.portfolio:
            if symbol.lower() in title.lower() or symbol.lower() in description.lower():
                portfolio_relevance += 1

        # Normalize the score
        impact_score = sentiment_score * (1 + portfolio_relevance)  # Add 1 to give base sentiment some weight
        return impact_score

    def personalize_feed(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:  # Added type hints
        """
        Personalizes the news feed based on user preferences and sentiment analysis.

        Args:
            articles (list): List of news articles to personalize.

        Returns:
            list: Personalized and ranked news articles.
        """
        personalized_articles: List[Dict[str, Any]] = []  # Added type hint

        # Rank articles based on sentiment analysis and impact score
        for article in articles:
            impact_score = self.analyze_impact(article)
            article['impact_score'] = impact_score  # Storing it in the article dict
            article['sentiment_score'] = self.analyze_sentiment(article)  # Storing it
            personalized_articles.append(article)

        # Sort articles by impact score (highest to lowest)
        personalized_articles.sort(key=lambda x: x.get('impact_score', 0), reverse=True)

        return personalized_articles

    def send_alerts(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Identifies articles meeting alert criteria, prints alerts for new ones,
        and returns all articles that met criteria in this batch.

        Args:
            articles (list): List of personalized news articles.

        Returns:
            List[Dict[str, Any]]: Articles from the input that met alert criteria.
        """
        alerting_thresholds = self.config.get('alerting_thresholds', {})
        positive_threshold = alerting_thresholds.get('positive_impact', 0.5)
        negative_threshold = alerting_thresholds.get('negative_impact', -0.5)

        alert_worthy_articles_in_batch = []

        for article in articles:
            impact_score = article.get('impact_score', 0)
            title = article.get('title', 'No Title')
            article_url = article.get('link', article.get('url'))  # Prefer 'link', fallback to 'url'

            is_alert_worthy = False
            alert_type = ""

            if impact_score > positive_threshold:
                is_alert_worthy = True
                alert_type = "Impactful news"
            elif impact_score < negative_threshold:
                is_alert_worthy = True
                alert_type = "Negative news"

            if is_alert_worthy:
                alert_worthy_articles_in_batch.append(article)
                # Use a more robust check for article_url's presence and content
                if article_url and isinstance(article_url, str) and article_url.strip():
                    if article_url not in self.seen_alert_urls:
                        print(f"ALERT: {alert_type} - {title}\nScore: {impact_score:.2f}\nLink: {article_url}")
                        self.seen_alert_urls.add(article_url)
                else:  # If no valid URL, alert based on title (might lead to duplicates if titles aren't unique)
                    temp_article_id = title
                    if temp_article_id not in self.seen_alert_urls:
                        print(f"ALERT (no URL): {alert_type} - {title}\nScore: {impact_score:.2f}")
                        self.seen_alert_urls.add(temp_article_id)

        return alert_worthy_articles_in_batch

    # --- AgentBase Methods Implementation ---

    async def execute(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """
        Main execution method for the NewsBot agent.
        Aggregates, personalizes, alerts on news, and generates a report.
        """
        print(f"NewsBot executing cycle at {datetime.now(timezone.utc).isoformat()}")

        news_articles = self.aggregate_news()
        personalized_feed = self.personalize_feed(news_articles)

        self.send_alerts(personalized_feed)

        self.aggregated_news = personalized_feed

        analysis_report = None
        # Use a config flag to enable/disable reporting, defaulting to True for now
        if self.config.get('enable_analysis_reporting', True) and personalized_feed:
            summary = await self.summarize_articles(personalized_feed[:5])
            critical_analysis = self.perform_critical_analysis(personalized_feed[:10])
            actionable_insights = self.draw_conclusions(critical_analysis)
            analysis_report = self.generate_report(
                personalized_feed[:5], summary, critical_analysis, actionable_insights)
            # Removed automatic printing of report from here to avoid clutter during monitoring.
            # It will be printed by the standalone runner if requested.
        return {
            'personalized_feed': personalized_feed,
            'analysis_report': analysis_report
        }

    async def monitor_for_critical_news(self, duration_minutes: int = 5, interval_seconds: int = 60) -> List[Dict[str, Any]]:
        """Monitors for critical news updates for a specified duration and interval."""
        print(f"Starting news monitoring for {duration_minutes} minutes, checking every {interval_seconds} seconds.")
        # Using a dictionary to store unique articles by URL to avoid duplicates.
        all_critical_news_session: Dict[str, Dict[str, Any]] = {}

        loop = asyncio.get_event_loop()
        end_time = loop.time() + duration_minutes * 60

        while loop.time() < end_time:
            print(f"Monitoring... running news check at {datetime.now(timezone.utc).isoformat()}")

            results_dict = await self.execute()  # This already calls send_alerts internally

            # Identify articles that meet alerting criteria from the current batch
            alerting_thresholds = self.config.get('alerting_thresholds', {})
            positive_threshold = alerting_thresholds.get('positive_impact', 0.5)
            negative_threshold = alerting_thresholds.get('negative_impact', -0.5)

            if results_dict.get('personalized_feed'):
                for article in results_dict['personalized_feed']:
                    impact_score = article.get('impact_score', 0.0)  # Default to 0.0 if not present
                    if impact_score > positive_threshold or impact_score < negative_threshold:
                        article_url = article.get('link', article.get('url'))
                        # Ensure URL is valid and prefer it as key for uniqueness
                        unique_key = article_url if (article_url and isinstance(article_url, str) and article_url.strip()) else article.get(
                            'title', str(random.random()))  # Fallback key

                        if unique_key not in all_critical_news_session:
                            all_critical_news_session[unique_key] = article

            remaining_time = end_time - loop.time()
            if remaining_time > interval_seconds:
                print(f"Monitoring... next check in {interval_seconds}s.")
                await asyncio.sleep(interval_seconds)
            elif remaining_time > 0:  # Sleep for the remaining time if less than interval
                await asyncio.sleep(remaining_time)
            else:  # No time left
                break

        print("Monitoring finished.")
        return list(all_critical_news_session.values())

    async def summarize_articles(self, articles: List[Dict[str, Any]]) -> str:
        """Generates a summary from a list of articles."""
        if not articles:
            return "No articles provided for summarization."

        if self.summarizer_tokenizer and self.summarizer_model:
            try:
                text_to_summarize = " ".join([
                    article.get('description', article.get('title', ''))
                    for article in articles if article.get('description', article.get('title', ''))
                ])
                if not text_to_summarize.strip():
                    return "No content available in provided articles for summarization."

                inputs = self.summarizer_tokenizer(
                    text_to_summarize,
                    return_tensors="pt",
                    max_length=1024,  # Max input length for BART models
                    truncation=True,
                    padding="longest"
                )
                summary_ids = self.summarizer_model.generate(
                    inputs.input_ids,
                    num_beams=4,
                    max_length=150,  # Desired summary length
                    early_stopping=True
                )
                summary = self.summarizer_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
                return summary
            except Exception as e:
                print(f"Error during model-based summarization: {e}. Falling back to basic summarization.")

        # Fallback to basic summarization (first sentence of top N articles)
        print("Warning: Summarization model not available or failed. Using fallback (first sentence concatenation).")
        summary_parts = []
        for i, article in enumerate(articles[:3]):  # Top 3 articles for fallback summary
            text = article.get('description', article.get('title', ''))
            if text:
                try:
                    # Ensure 'punkt' is downloaded for nltk.sent_tokenize
                    # This check is also in __init__, but good to have here if that failed silently.
                    nltk.data.find('tokenizers/punkt')
                except nltk.downloader.DownloadError:
                    print("NLTK 'punkt' not found. Cannot perform fallback summarization.")
                    return "Summarization fallback failed: NLTK 'punkt' missing."

                sentences = nltk.sent_tokenize(text)
                if sentences:
                    summary_parts.append(sentences[0])

        return " ".join(summary_parts) if summary_parts else "No content for fallback summary."

    def perform_critical_analysis(self, articles: List[Dict[str, Any]]) -> str:
        """Performs a placeholder critical analysis based on sentiment and keywords."""
        if not articles:
            return "No articles for critical analysis."

        analysis_points = []

        # 1. Conflicting sentiment for portfolio items
        # Simplified: Check if any article about a portfolio item has strong pos/neg sentiment
        # A more advanced version would track sentiment per item across articles.
        portfolio_symbols = [str(s).lower() for s in self.portfolio.get(
            'stocks', []) + self.portfolio.get('crypto', [])]

        conflicting_sentiments = {}  # stock_symbol: [sentiments]

        for article in articles:
            title = article.get('title', '').lower()
            description = article.get('description', '').lower()
            content = title + " " + description
            sentiment = article.get('sentiment_score', 0.0)

            for symbol in portfolio_symbols:
                if symbol in content:
                    if symbol not in conflicting_sentiments:
                        conflicting_sentiments[symbol] = []
                    conflicting_sentiments[symbol].append(sentiment)

        for symbol, sentiments in conflicting_sentiments.items():
            has_positive = any(s > 0.5 for s in sentiments)
            has_negative = any(s < -0.5 for s in sentiments)
            if has_positive and has_negative:
                analysis_points.append(
                    f"Conflicting sentiment found for portfolio item: {symbol.upper()}. Articles show both positive and negative views.")

        # 2. Predefined risk/opportunity keywords
        risk_keywords = ["warning", "concern", "risk", "volatile", "downturn", "bubble"]
        opportunity_keywords = ["breakthrough", "opportunity", "growth", "bullish", "upward", "promising"]

        found_risks = set()
        found_opportunities = set()

        for article in articles:
            content = (article.get('title', '') + " " + article.get('description', '')).lower()
            for r_keyword in risk_keywords:
                if r_keyword in content:
                    found_risks.add(r_keyword)
            for o_keyword in opportunity_keywords:
                if o_keyword in content:
                    found_opportunities.add(o_keyword)

        if found_risks:
            analysis_points.append(f"Potential risks indicated by keywords: {', '.join(list(found_risks))}.")
        if found_opportunities:
            analysis_points.append(
                f"Potential opportunities indicated by keywords: {', '.join(list(found_opportunities))}.")

        return "\n- ".join(analysis_points) if analysis_points else "No specific critical points identified from the top articles."

    def draw_conclusions(self, critical_analysis_text: str) -> str:
        """Draws placeholder actionable insights from critical analysis."""
        insights = []
        if not critical_analysis_text or critical_analysis_text == "No specific critical points identified from the top articles.":
            return "No specific actionable insights based on current analysis."

        if "conflicting sentiment" in critical_analysis_text.lower():
            # Extract stock symbols if possible (very basic)
            parts = critical_analysis_text.split(":")
            if len(parts) > 1 and "portfolio item" in parts[0].lower():
                stock_symbol = parts[1].split(".")[0].strip()
                insights.append(f"Review {stock_symbol.upper()} due to conflicting reports and sentiment.")
            else:
                insights.append("Review portfolio items with conflicting sentiment reports.")

        if "potential risks indicated by keywords" in critical_analysis_text.lower():
            insights.append("Monitor news closely for identified potential risks.")

        if "potential opportunities indicated by keywords" in critical_analysis_text.lower():
            insights.append("Explore potential opportunities highlighted in the news.")

        return "\n- ".join(insights) if insights else "General market awareness recommended based on analysis."

    def generate_report(self, articles: List[Dict[str, Any]], summary: str, critical_analysis: str, actionable_insights: str) -> str:
        """Formats the analysis into a structured report string."""
        report_parts = [
            "--- News Analysis Report ---",
            "\nOverall Summary:",
            summary,
            "\nCritical Analysis:",
            f"- {critical_analysis}",  # Assuming critical_analysis is already formatted with newlines or is a single block
            "\nActionable Insights:",
            f"- {actionable_insights}",  # Same assumption for actionable_insights
            "\nSources (Top Articles Used for Summary/Analysis):"
        ]

        for i, article in enumerate(articles[:5]):  # List sources for top 5 articles
            title = article.get('title', 'N/A')
            link = article.get('link', article.get('url', 'N/A'))
            sentiment = article.get('sentiment_score', 0.0)
            impact = article.get('impact_score', 0.0)
            report_parts.append(
                f"  {i+1}. Title: {title}\n     Link: {link}\n     Sentiment: {sentiment:.2f}, Impact: {impact:.2f}"
            )

        report_parts.append("\n--- End of Report ---")
        return "\n".join(report_parts)

    def get_skill_schema(self) -> Dict[str, Any]:
        """
        Defines the NewsBot's skills for MCP.
        """
        schema = super().get_skill_schema()  # Get base schema
        schema["description"] = self.config.get(
            "description", "Monitors news, aggregates, personalizes, and alerts on relevant events.")
        schema["skills"] = [
            {
                "name": "get_latest_news",
                "description": "Retrieves the latest personalized news articles.",
                "parameters": []  # No specific input parameters for this general fetch
            },
            {
                "name": "get_news_for_topic",
                "description": "Retrieves news articles for a specific topic.",
                "parameters": [
                    {"name": "topic", "type": "string", "description": "The topic to search news for."}
                ]
            },
            {
                "name": "web_search",
                "description": "Performs a web search for a given query and returns relevant articles.",
                "parameters": [  # Changed from 'inputs' to 'parameters' for consistency with other skills
                    {"name": "query", "type": "string", "description": "The search query."}
                ],
                # "outputs" key is not standard for my current get_skill_schema structure.
            },
            {
                "name": "generate_news_analysis_report",
                "description": "Generates a news analysis report with summary, critical analysis, and actionable insights from the latest personalized news.",
                "parameters": [],  # Uses internally aggregated news
                # "outputs": [
                #     {"name": "report", "type": "string", "description": "The generated news analysis report."}
                # ]
            },
            {
                "name": "start_news_monitoring",
                "description": "Starts monitoring for critical news updates for a specified duration and interval.",
                "parameters": [
                    {"name": "duration_minutes", "type": "integer",
                        "description": "Duration to monitor in minutes.", "optional": True, "default": 5},
                    {"name": "interval_seconds", "type": "integer",
                        "description": "Interval between checks in seconds.", "optional": True, "default": 60}
                ],
                # "outputs": [ # If we were to define outputs for schema
                #     {"name": "critical_articles", "type": "list", "description": "A list of unique critical articles found during the monitoring session."}
                # ]
            }
        ]
        return schema

    async def search_web(self, query: str) -> List[Dict[str, Any]]:
        """
        Performs a web search for a given query using a placeholder.
        In a real scenario, this would use an actual search engine API.
        """
        if not self.search_api_key:
            print("Warning: Search API key not available. Web search cannot be performed.")
            return []

        print(f"INFO: Performing web search for: \"{query}\" using a placeholder API.")

        # Placeholder for actual search API call
        mock_results: List[Dict[str, Any]] = []
        try:
            # Simulate API call and result transformation
            current_time_iso = datetime.now(timezone.utc).isoformat()
            mock_results = [
                {
                    'title': f"Mock Search Result 1 for '{query}'",
                    'link': f"http://example.com/search?q={query.replace(' ', '+')}&result=1",
                    'description': f"This is a mock description for the first web search result related to '{query}'.",
                    'published_at': current_time_iso,
                    'source': {'name': 'Web Search Result'}
                },
                {
                    'title': f"Mock Search Result 2 for '{query}'",
                    'link': f"http://example.com/search?q={query.replace(' ', '+')}&result=2",
                    'description': f"Another detailed mock description for the second web search result concerning '{query}'.",
                    'published_at': current_time_iso,
                    'source': {'name': 'Web Search Result'}
                }
            ]
            # Simulate some processing delay if needed
            # await asyncio.sleep(0.1)
        except Exception as e:
            print(f"Error during placeholder web search for '{query}': {e}")
            return []  # Return empty list on error

        return mock_results

    async def receive_message(self, sender_agent: str, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Handles incoming A2A messages.
        Example: Another agent might request news for a specific topic.
        """
        print(f"NewsBot received message from {sender_agent}: {message}")
        action = message.get("action")
        if action == "get_news_for_topic":
            topic = message.get("topic")
            if topic:
                # Simplified: re-run aggregation focusing on this topic.
                # A more advanced implementation would filter existing news or fetch specifically.
                original_topics = self.user_preferences.get('topics', [])
                self.user_preferences['topics'] = [topic]  # Temporarily override

                news_articles = self.aggregate_news()
                personalized_feed = self.personalize_feed(news_articles)

                self.user_preferences['topics'] = original_topics  # Restore
                return {"status": "success", "articles": personalized_feed[:10]}  # Return top 10
            else:
                return {"status": "error", "message": "Topic not provided"}

        return await super().receive_message(sender_agent, message)  # Default handling


if __name__ == "__main__":
    import argparse
    import json
    import asyncio
    import os

    def load_json_arg(arg_value: str, arg_name: str) -> Any:
        """Loads a JSON argument, trying as a filepath first, then as a string."""
        if arg_value is None:
            return None
        try:
            if os.path.exists(arg_value):
                with open(arg_value, 'r') as f:
                    return json.load(f)
            else:
                return json.loads(arg_value)
        except FileNotFoundError:
            print(f"Error: File not found for argument --{arg_name}: {arg_value}")
            exit(1)
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON for argument --{arg_name}. Content: {arg_value}")
            exit(1)
        except Exception as e:
            print(f"An unexpected error occurred while loading argument --{arg_name}: {e}")
            exit(1)

    parser = argparse.ArgumentParser(description="Run NewsBot Agent Standalone")
    parser.add_argument(
        "--user-preferences",
        required=True,
        help="User preferences as a JSON string or path to a JSON file. E.g., '{\"topics\": [\"finance\", \"crypto\"]}'"
    )
    parser.add_argument(
        "--news-api-key",
        required=True,
        help="API key for the general news aggregation service (NewsAPI)."
    )
    parser.add_argument(
        "--portfolio",
        required=True,
        help="User's financial portfolio as a JSON string or path to a JSON file. E.g., '{\"stocks\": [\"AAPL\"], \"crypto\": [\"BTC\"]}'"
    )
    parser.add_argument(
        "--custom-api-sources",
        required=False,
        default="[]",
        help="Optional list of custom API sources (JSON string or path to JSON file). E.g., '[{\"name\": \"MyNews\", \"url\": \"http://my.news/api\"}]'"
    )
    parser.add_argument(
        "--search-api-key",
        required=False,
        help="API key for the web search service (placeholder)."
    )
    parser.add_argument(
        "--search-query",
        required=False,
        help="If provided, performs a web search with this query and prints results."
    )
    parser.add_argument(
        "--generate-report",
        action="store_true",
        help="If provided, generates and prints an analysis report instead of just the article list."
    )
    parser.add_argument(
        "--monitor-news",
        action="store_true",
        help="If provided, starts news monitoring for a specified duration."
    )
    parser.add_argument(
        "--monitor-duration",
        type=int,
        default=5,
        help="Duration in minutes for news monitoring (default: 5)."
    )
    parser.add_argument(
        "--monitor-interval",
        type=int,
        default=60,
        help="Interval in seconds between monitoring checks (default: 60)."
    )

    args = parser.parse_args()

    # Ensure NLTK 'punkt' is available for summarization fallback
    # This is also in __init__ for robustness if NewsBot is used as a library
    try:
        nltk.data.find('tokenizers/punkt')
    except nltk.downloader.DownloadError:
        print("NLTK 'punkt' not found for standalone runner. Downloading...")
        nltk.download('punkt', quiet=True)

    user_preferences_data = load_json_arg(args.user_preferences, "user-preferences")
    portfolio_data = load_json_arg(args.portfolio, "portfolio")
    custom_api_sources_data = load_json_arg(args.custom_api_sources, "custom-api-sources")

    # Construct agent_config for NewsBot
    agent_config = {
        "persona": "Standalone NewsBot",
        "description": "NewsBot running in standalone mode.",
        "expertise": ["news aggregation", "event detection", "information filtering", "sentiment analysis", "web search"],
        "user_preferences": user_preferences_data,
        "news_api_key": args.news_api_key,
        "search_api_key": args.search_api_key,  # Added search_api_key to config
        "portfolio": portfolio_data,
        "user_api_sources": custom_api_sources_data,
        "alerting_thresholds": {
            "positive_impact": 0.6,
            "negative_impact": -0.6
        }
    }

    news_bot = NewsBot(config=agent_config, kernel=None)

    if args.monitor_news:
        # Make sure monitor_news is mutually exclusive with search and regular execute for CLI simplicity
        if args.search_query or args.generate_report:
            print("Error: --monitor-news cannot be used with --search-query or --generate-report in this standalone runner.")
            exit(1)
        print(
            f"\n--- Starting News Monitoring (Duration: {args.monitor_duration}m, Interval: {args.monitor_interval}s) ---")
        try:
            critical_articles_found = asyncio.run(
                news_bot.monitor_for_critical_news(
                    duration_minutes=args.monitor_duration,
                    interval_seconds=args.monitor_interval
                )
            )
            print("\n--- All Unique Critical Articles Found During Monitoring Session ---")
            if critical_articles_found:
                for i, article_item in enumerate(critical_articles_found):
                    print(f"\nCritical Article {i+1}:")
                    print(f"  Title: {article_item.get('title', 'N/A')}")
                    print(f"  Link: {article_item.get('link', article_item.get('url', 'N/A'))}")
                    print(f"  Source: {article_item.get('source', {}).get('name', 'N/A')}")
                    print(f"  Published At: {article_item.get('published_at', 'N/A')}")
                    sentiment_score = article_item.get('sentiment_score', 0.0)
                    impact_score = article_item.get('impact_score', 0.0)
                    print(f"  Sentiment Score: {float(sentiment_score):.2f}")  # Ensure float for formatting
                    print(f"  Impact Score: {float(impact_score):.2f}")   # Ensure float for formatting
            else:
                print("No new critical articles identified during the monitoring session.")
        except Exception as e:
            print(f"An error occurred during news monitoring: {e}")

    elif args.search_query:
        if args.generate_report:  # Ensure search and report are not combined if not desired
            print("Error: --search-query and --generate-report are mutually exclusive in this runner if not monitoring.")
            exit(1)
        print(f"\n--- Performing Web Search for: \"{args.search_query}\" ---")
        try:
            search_results = asyncio.run(news_bot.search_web(args.search_query))
            if search_results:
                for i, article_res in enumerate(search_results):
                    print(f"\nSearch Result {i+1}:")
                    print(f"  Title: {article_res.get('title', 'N/A')}")
                    print(f"  Link: {article_res.get('link', 'N/A')}")
                    print(f"  Source: {article_res.get('source', {}).get('name', 'N/A')}")
                    print(f"  Published At: {article_res.get('published_at', 'N/A')}")
                    if article_res.get('description'):
                        print(f"  Description: {article_res.get('description')[:150]}...")
            else:
                print("No search results returned or search_api_key not provided.")
        except Exception as e:
            print(f"An error occurred during web search: {e}")
    else:  # Default to aggregation/personalization and optional report (if not monitoring or searching)
        run_mode_message = "\n--- Running Full NewsBot Aggregation, Personalization"
        if args.generate_report:
            run_mode_message += " & Reporting"
        run_mode_message += " ---"
        print(run_mode_message)

        try:
            results = asyncio.run(news_bot.execute())
            personalized_feed_result = results.get('personalized_feed')
            analysis_report_result = results.get('analysis_report')

            if args.generate_report:
                if analysis_report_result:
                    print("\n--- Analysis Report ---")
                    print(analysis_report_result)
                else:  # Should only happen if personalized_feed was empty
                    print("\nNo analysis report was generated (e.g., no articles to analyze).")
            else:
                print("\n--- Personalized News Feed (Top Articles) ---")
                if personalized_feed_result:
                    for i, article_item in enumerate(personalized_feed_result[:10]):
                        print(f"\nArticle {i+1}:")
                        print(f"  Title: {article_item.get('title', 'N/A')}")
                        print(f"  Link: {article_item.get('link', article_item.get('url', 'N/A'))}")
                        print(f"  Source: {article_item.get('source', {}).get('name', 'N/A')}")
                        print(f"  Published At: {article_item.get('published_at', 'N/A')}")
                        sentiment_score = article_item.get('sentiment_score', 0.0)
                        impact_score = article_item.get('impact_score', 0.0)
                        print(f"  Sentiment Score: {float(sentiment_score):.2f}")
                        print(f"  Impact Score: {float(impact_score):.2f}")
                        if article_item.get('description'):
                            print(f"  Description: {article_item.get('description')[:150]}...")
                else:
                    print("No news articles processed or returned in personalized feed.")

        except Exception as e:
            print(f"An error occurred during NewsBot execution: {e}")
            exit(1)

    print("\nNewsBot standalone execution finished.")
