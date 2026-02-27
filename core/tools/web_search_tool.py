import logging
import requests
from bs4 import BeautifulSoup
from typing import Optional, List
from duckduckgo_search import DDGS
from googlesearch import search as google_search
from semantic_kernel.functions.kernel_function_decorator import kernel_function
from core.tools.base_tool import BaseTool

class WebSearchTool(BaseTool):
    name: str = "web_search"
    description: str = "Performs a web search for a given query or fetches content from a direct URL."

    def _get_parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query. Used if URL is not provided."},
                "url": {"type": "string", "description": "The direct URL to fetch content from. Takes precedence over query."}
            },
            "required": []
        }

    @kernel_function(
        name="fetch_web_content",
        description="Fetches text content from a given URL or performs a search query."
    )
    async def execute(self, query: str = None, url: str = None) -> str:
        """
        Executes a web search or fetches a URL.
        """
        if not url and not query:
            return "Error: Either a query or a direct URL must be provided."

        if url:
            return self._fetch_url(url)

        if query:
            logging.info(f"WebSearchTool: Searching for '{query}'...")
            try:
                # 1. Try DuckDuckGo
                with DDGS() as ddgs:
                    results = list(ddgs.text(query, max_results=1))
                    if results:
                        top_url = results[0]['href']
                        logging.info(f"WebSearchTool: Found URL via DDG: {top_url}")
                        return self._fetch_url(top_url)
            except Exception as e:
                logging.warning(f"WebSearchTool: DuckDuckGo failed: {e}")

            try:
                # 2. Fallback to Google Search
                # googlesearch.search returns an iterator of URLs
                results = list(google_search(query, num_results=1))
                if results:
                    top_url = results[0]
                    logging.info(f"WebSearchTool: Found URL via Google: {top_url}")
                    return self._fetch_url(top_url)
            except Exception as e:
                logging.error(f"WebSearchTool: Google Search failed: {e}")

            return "Error: No search results found or search failed."

        return "Error: Unexpected state."

    def _fetch_url(self, url: str) -> str:
        logging.info(f"WebSearchTool: Fetching content from {url}")
        try:
            # Add headers to mimic a browser
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'html.parser')

            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()

            text = soup.get_text()

            # Break into lines and remove leading/trailing space on each
            lines = (line.strip() for line in text.splitlines())
            # Break multi-headlines into a line each
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            # Drop blank lines
            text = '\n'.join(chunk for chunk in chunks if chunk)

            return text[:5000]  # Return first 5000 chars

        except Exception as e:
            logging.error(f"WebSearchTool: Error fetching URL {url}: {e}")
            return f"Error: Could not fetch content from {url}. Details: {str(e)}"
