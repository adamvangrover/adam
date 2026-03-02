import logging
import requests
from bs4 import BeautifulSoup
from typing import Any, List, Dict
from core.tools.base_tool import BaseTool
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

logger = logging.getLogger("WebSearchTool")

class WebSearchTool(BaseTool):
    name: str = "web_search"
    description: str = "Performs a web search using DuckDuckGo (primary) or Google (fallback) and fetches content."

    def _get_parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query."},
                "url": {"type": "string", "description": "Direct URL to fetch (optional, bypasses search)."},
                "num_results": {"type": "integer", "description": "Number of results to return (default 3)."}
            },
            "required": ["query"]
        }

    @kernel_function(
        name="execute",
        description="Executes a web search or fetches a URL."
    )
    def execute(self, query: str = None, url: str = None, num_results: int = 3) -> str:
        """
        Executes a web search or content fetch.
        """
        if url:
            return self.fetch_web_content(url)
        
        if not query:
            return "Error: Query or URL must be provided."

        logger.info(f"Searching for: {query}")
        results = []

        # 1. DuckDuckGo
        if DDGS_AVAILABLE:
            try:
                with DDGS() as ddgs:
                    # duckduckgo-search v4+ returns a generator of dicts
                    search_gen = ddgs.text(query, max_results=num_results)
                    for r in search_gen:
                        results.append(r)
            except Exception as e:
                logger.error(f"DuckDuckGo search failed: {e}")

        # 2. Google Fallback
        if not results and GOOGLE_AVAILABLE:
            try:
                # googlesearch-python returns urls
                g_results = google_search(query, num_results=num_results, advanced=True)
                for r in g_results:
                    results.append({"href": r.url, "title": r.title, "body": r.description})
            except Exception as e:
                logger.error(f"Google search failed: {e}")

        if not results:
            return "No search results found."

        # Format results
        output = f"Search Results for '{query}':\n\n"
        for i, res in enumerate(results):
            link = res.get("href") or res.get("link")
            title = res.get("title")
            snippet = res.get("body") or res.get("snippet")
            
            output += f"{i+1}. {title}\n   URL: {link}\n   Snippet: {snippet}\n\n"
            
            # Fetch content of top result if it's a direct query
            if i == 0 and num_results == 1:
                 content = self.fetch_web_content(link)
                 output += f"--- Content of Top Result ---\n{content[:2000]}...\n"

        return output

    def fetch_web_content(self, url: str) -> str:
        """
        Fetches and strips HTML from a URL.
        """
        try:
            logger.info(f"Fetching content from: {url}")
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove scripts and styles
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.extract()
                
            text = soup.get_text(separator='\n')
            
            # Clean whitespace
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            clean_text = '\n'.join(chunk for chunk in chunks if chunk)
            
            return clean_text
            
        except Exception as e:
            logger.error(f"Error fetching content from {url}: {e}")
            return f"Error fetching content: {str(e)}"
