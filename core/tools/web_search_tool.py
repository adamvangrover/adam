import logging
from typing import Any
from core.tools.base_tool import BaseTool

import requests
from bs4 import BeautifulSoup
from semantic_kernel.functions.kernel_function_decorator import kernel_function

try:
    from duckduckgo_search import DDGS
    DDGS_AVAILABLE = True
except ImportError:
    DDGS_AVAILABLE = False

try:
    from googlesearch import search as google_search
    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False


class WebSearchTool(BaseTool):
    name: str = "web_search"
    description: str = "Performs a web search for a given query or fetches content from a direct URL."

    def _get_parameters_schema(self) -> dict:
        # This schema is for the BaseTool's get_schema method.
        # Semantic Kernel will infer from the @kernel_function decorated method's signature.
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query. Used if URL is not provided."},
                "url": {"type": "string", "description": "The direct URL to fetch content from. Takes precedence over query."}
            },
            "required": []
        }

    @kernel_function(
        name="execute",
        description="Executes a web search or fetches a URL."
    )
    async def execute(self, query: str = None, url: str = None, num_results: int = 3) -> str:
        """
        Executes a web search or content fetch.
        """
        if url:
            return await self.fetch_web_content(url)

        if not query:
            return "Error: Query or URL must be provided."

        logging.info(f"Searching for: {query}")
        results = []

        # 1. DuckDuckGo
        if DDGS_AVAILABLE:
            try:
                with DDGS() as ddgs:
                    search_gen = ddgs.text(query, max_results=num_results)
                    for r in search_gen:
                        results.append(r)
            except Exception as e:
                logging.error(f"DuckDuckGo search failed: {e}")

        # 2. Google Fallback
        if not results and GOOGLE_AVAILABLE:
            try:
                g_results = google_search(query, num_results=num_results, advanced=True)
                for r in g_results:
                    results.append({"href": r.url, "title": r.title, "body": r.description})
            except Exception as e:
                logging.error(f"Google search failed: {e}")

        if not results:
            return "No search results found."

        output = f"Search Results for '{query}':\n\n"
        for i, res in enumerate(results):
            link = res.get("href") or res.get("link")
            title = res.get("title")
            snippet = res.get("body") or res.get("snippet")

            output += f"{i+1}. {title}\n   URL: {link}\n   Snippet: {snippet}\n\n"

            if i == 0 and num_results == 1:
                 content = await self.fetch_web_content(link)
                 output += f"--- Content of Top Result ---\n{content[:2000]}...\n"

        return output

    async def fetch_web_content(self, url: str) -> str:
        """
        Fetches and strips HTML from a URL.
        """
        try:
            logging.info(f"Fetching content from: {url}")
            headers = {
                'User-Agent': (
                    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                    'AppleWebKit/537.36 (KHTML, like Gecko) '
                    'Chrome/91.0.4472.124 Safari/537.36'
                )
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'html.parser')
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.extract()

            text = soup.get_text(separator='\n')
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            clean_text = '\n'.join(chunk for chunk in chunks if chunk)

            return clean_text

        except Exception as e:
            logging.error(f"Error fetching content from {url}: {e}")
            return f"Error fetching content: {str(e)}"
