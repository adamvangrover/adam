# core/agents/lexica_agent.py

import requests
import logging
import asyncio
from typing import Dict, Any, Optional
from core.agents.agent_base import AgentBase

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LexicaAgent(AgentBase):
    """
    Agent responsible for information retrieval from various sources.
    """

    def __init__(self, config: Dict[str, Any], kernel: Optional[Any] = None):
        """
        Initializes the Lexica Agent.

        Args:
            config (dict): Configuration dictionary.
            kernel (Optional[Any]): Semantic Kernel instance.
        """
        super().__init__(config, kernel=kernel)
        self.data_sources = self.config.get('data_sources', {})
        self.api_keys = self.config.get('api_keys', {})

    async def execute(self, *args, **kwargs):
        """
        Retrieves information based on a query.

        Args:
            query (str): The query or topic for information retrieval via kwargs.

        Returns:
            dict: A dictionary containing relevant information and sources.
        """
        query = kwargs.get('query')
        if not query:
            return {"error": "No query provided."}

        logger.info(f"LexicaAgent retrieving information for: {query}")

        # Execute retrieval tasks concurrently
        # Note: Actual HTTP calls should be async, here we wrap synchronous calls
        loop = asyncio.get_running_loop()

        # 1. Web Search
        web_task = loop.run_in_executor(None, self.search_web, query)

        # 2. News Articles
        news_task = loop.run_in_executor(None, self.get_news_articles, query)

        # 3. Financial Data
        finance_task = loop.run_in_executor(None, self.get_financial_data, query)

        web_results, news_articles, financial_data = await asyncio.gather(web_task, news_task, finance_task)

        information = {
            'web_results': web_results,
            'news_articles': news_articles,
            'financial_data': financial_data,
        }
        return information

    def search_web(self, query):
        """
        Searches the web using a search engine API.
        """
        results = []
        api_key = self.api_keys.get('google_search_api_key')
        engine_id = self.api_keys.get('google_search_engine_id')

        if not api_key or not engine_id:
            logger.warning("Google Search API keys missing.")
            return []

        url = f"https://www.googleapis.com/customsearch/v1?key={api_key}&cx={engine_id}&q={query}"
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                results = [
                    {"title": item.get('title'), "link": item.get('link'), "snippet": item.get('snippet')}
                    for item in data.get('items', [])
                ]
            else:
                logger.error(f"Error searching the web: {response.status_code}")
        except Exception as e:
            logger.error(f"Web search exception: {e}")

        return results

    def get_news_articles(self, query):
        """
        Fetches news articles using a news aggregator API.
        """
        articles = []
        api_key = self.api_keys.get('news_api_key')

        if not api_key:
            logger.warning("NewsAPI key missing.")
            return []

        url = f"https://newsapi.org/v2/everything?q={query}&apiKey={api_key}"
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                articles = [
                    {"title": item.get('title'), "link": item.get('url'), "source": item.get('source', {}).get('name')}
                    for item in data.get('articles', [])
                ]
            else:
                logger.error(f"Error fetching news articles: {response.status_code}")
        except Exception as e:
            logger.error(f"News fetch exception: {e}")

        return articles

    def get_financial_data(self, query):
        """
        Retrieves financial data from a database or API.
        """
        data = {}
        api_key = self.api_keys.get('iex_api_key')

        if not api_key:
            logger.warning("IEX Cloud API key missing.")
            return {}

        url = f"https://cloud.iexapis.com/stable/stock/{query}/quote?token={api_key}"
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                data = response.json()
            else:
                logger.error(f"Error fetching financial data: {response.status_code}")
        except Exception as e:
            logger.error(f"Financial data fetch exception: {e}")

        return data

if __name__ == "__main__":
    config = {'api_keys': {'news_api_key': 'mock'}}
    agent = LexicaAgent(config)
    async def main():
        res = await agent.execute(query="Apple")
        print(res)
    asyncio.run(main())
