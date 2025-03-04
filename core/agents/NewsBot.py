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

# Initialize NLTK sentiment analyzer
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

# Define the "NewsBot" class (a more concise name for the agent)
class NewsBot:
    def __init__(self, user_preferences, news_api_key, portfolio, user_api_sources=None):
        """
        Initializes the NewsBot Agent.
        
        Args:
            user_preferences (dict): A dictionary containing user preferences.
            news_api_key (str): API key for the general news aggregation service.
            portfolio (dict): A dictionary containing the user's financial holdings.
            user_api_sources (list, optional): List of custom API sources provided by the user.
        """
        self.user_preferences = user_preferences
        self.news_api_key = news_api_key
        self.portfolio = portfolio
        self.user_api_sources = user_api_sources or []
        
        self.cg = CoinGeckoAPI()  # For crypto-related news
        self.aggregated_news = []
        self.user_interactions = defaultdict(int)  # Keep track of user's news interactions
        
        # Load user-defined news sources
        self.custom_news_sources = self.load_custom_sources()

    def load_custom_sources(self):
        """Load custom news APIs provided by the user."""
        custom_sources = {}
        for source in self.user_api_sources:
            custom_sources[source['name']] = source['url']
        return custom_sources

    def aggregate_news(self):
        """
        Aggregates news articles from various sources (including custom user sources).
        
        Returns:
            list: Aggregated news articles
        """
        all_news = []

        # Collect crypto-related news
        if 'crypto' in self.user_preferences['topics']:
            all_news.extend(self.get_crypto_news())

        # Collect financial market-related news
        if 'finance' in self.user_preferences['topics']:
            all_news.extend(self.get_finance_news())

        # Collect stock market news
        if 'stocks' in self.user_preferences['topics']:
            all_news.extend(self.get_stock_news())

        # Collect commodities and treasury news
        if 'commodities' in self.user_preferences['topics']:
            all_news.extend(self.get_commodities_news())

        if 'treasuries' in self.user_preferences['topics']:
            all_news.extend(self.get_treasuries_news())

        # Collect FX news
        if 'forex' in self.user_preferences['topics']:
            all_news.extend(self.get_forex_news())

        # Integrate custom sources (user-defined)
        for source_name, source_url in self.custom_news_sources.items():
            all_news.extend(self.get_custom_news(source_url))

        # Filter news based on user's portfolio
        filtered_news = self.filter_news_by_portfolio(all_news)

        return filtered_news

    def get_crypto_news(self):
        """Fetch cryptocurrency news using the CoinGecko API."""
        return self.cg.get_trending_searches(language='en')

    def get_finance_news(self):
        """Fetch general finance news using NewsAPI."""
        url = "https://newsapi.org/v2/everything"
        params = {
            'q': 'finance',
            'apiKey': self.news_api_key,
            'language': 'en',
            'sortBy': 'relevancy'
        }
        response = requests.get(url, params=params)
        return response.json().get('articles', [])

    def get_stock_news(self):
        """Fetch stock-related news using an API or data source."""
        url = "https://newsapi.org/v2/everything"
        params = {
            'q': 'stocks',
            'apiKey': self.news_api_key,
            'language': 'en',
            'sortBy': 'relevancy'
        }
        response = requests.get(url, params=params)
        return response.json().get('articles', [])

    def get_commodities_news(self):
        """Fetch commodities-related news (gold, oil, etc.)."""
        url = "https://newsapi.org/v2/everything"
        params = {
            'q': 'commodities',
            'apiKey': self.news_api_key,
            'language': 'en',
            'sortBy': 'relevancy'
        }
        response = requests.get(url, params=params)
        return response.json().get('articles', [])

    def get_treasuries_news(self):
        """Fetch treasury bond-related news."""
        url = "https://newsapi.org/v2/everything"
        params = {
            'q': 'treasury bonds',
            'apiKey': self.news_api_key,
            'language': 'en',
            'sortBy': 'relevancy'
        }
        response = requests.get(url, params=params)
        return response.json().get('articles', [])

    def get_forex_news(self):
        """Fetch foreign exchange news."""
        url = "https://newsapi.org/v2/everything"
        params = {
            'q': 'forex',
            'apiKey': self.news_api_key,
            'language': 'en',
            'sortBy': 'relevancy'
        }
        response = requests.get(url, params=params)
        return response.json().get('articles', [])

    def get_custom_news(self, api_url):
        """Fetch custom news from user-provided sources."""
        try:
            response = requests.get(api_url)
            return response.json().get('articles', [])
        except Exception as e:
            print(f"Error fetching custom news from {api_url}: {e}")
            return []

    def filter_news_by_portfolio(self, news_articles):
        """
        Filters news articles based on portfolio holdings (stocks, crypto, etc.).
        
        Args:
            news_articles (list): List of news articles.
        
        Returns:
            list: Filtered list of news articles.
        """
        filtered_news = []
        for article in news_articles:
            for symbol in self.portfolio:
                if symbol.lower() in article['title'].lower() or symbol.lower() in article['description'].lower():
                    filtered_news.append(article)
                    break
        return filtered_news

    def analyze_sentiment(self, article):
        """Analyze the sentiment of a news article using VADER."""
        text = article['title'] + " " + article.get('description', '')
        sentiment_score = sia.polarity_scores(text)['compound']
        return sentiment_score

    def analyze_impact(self, article):
        """
        Calculates an 'impact score' based on sentiment, portfolio relevance, and topic.
        
        Args:
            article (dict): The news article to evaluate.
        
        Returns:
            float: The impact score of the article.
        """
        sentiment_score = self.analyze_sentiment(article)
        portfolio_relevance = 0

        # Increase impact score if it matches portfolio holdings
        for symbol in self.portfolio:
            if symbol.lower() in article['title'].lower() or symbol.lower() in article['description'].lower():
                portfolio_relevance += 1
        
        # Normalize the score
        impact_score = sentiment_score * portfolio_relevance
        return impact_score

    def personalize_feed(self, articles):
        """
        Personalizes the news feed based on user preferences and sentiment analysis.
        
        Args:
            articles (list): List of news articles to personalize.
        
        Returns:
            list: Personalized and ranked news articles.
        """
        personalized_articles = []

        # Rank articles based on sentiment analysis and impact score
        for article in articles:
            impact_score = self.analyze_impact(article)
            article['impact_score'] = impact_score
            article['sentiment_score'] = self.analyze_sentiment(article)
            personalized_articles.append(article)

        # Sort articles by impact score (highest to lowest)
        personalized_articles.sort(key=lambda x: x['impact_score'], reverse=True)

        return personalized_articles

    def send_alerts(self, articles):
        """
        Send alerts to users when critical news is detected based on user preferences.
        
        Args:
            articles (list): List of personalized news articles.
        """
        for article in articles:
            if article['impact_score'] > 0.5:  # High impact score (positive news)
                print(f"ALERT: Impactful news - {article['title']}\nScore: {article['impact_score']}")
            elif article['impact_score'] < -0.5:  # High impact negative news
                print(f"ALERT: Negative news - {article['title']}\nScore: {article['impact_score']}")

    def run(self):
        """Main function to run the news aggregator and personalization logic."""
        news_articles = self.aggregate_news()
        personalized_feed = self.personalize_feed(news_articles)
        self.send_alerts(personalized_feed)
        return personalized_feed

