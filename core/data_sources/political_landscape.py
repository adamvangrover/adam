"""
Political Landscape Loader

This module provides functionality to load political landscape data from various sources.
It is designed to be used by the RegulatoryComplianceAgent.
"""
import requests
from bs4 import BeautifulSoup
from typing import Dict, List, Optional
import datetime
import logging

# Configure logger
logger = logging.getLogger(__name__)

class PoliticalLandscapeLoader:
    """
    Loads political landscape data from external sources.
    Uses a combination of scraping and predefined data structures.
    """

    def __init__(self):
        self.sources = {
            "whitehouse_briefing": "https://www.whitehouse.gov/briefing-room/",
            "reuters_politics": "https://www.reuters.com/world/us/", # General US news/politics
        }
        # Fallback data in case of connection errors
        self.fallback_data = {
            "US": {
                "president": "Joe Biden",
                "party": "Democrat",
                "key_policies": [
                    "Infrastructure Investment and Jobs Act",
                    "Inflation Reduction Act",
                    "CHIPS and Science Act"
                ],
                "recent_developments": [
                    "Ongoing implementation of climate policies.",
                    "Discussions on federal budget and debt ceiling."
                ]
            }
        }

    def load_landscape(self) -> Dict:
        """
        Loads the political landscape.
        Currently focuses on US data, but can be extended.
        """
        landscape = {
            "US": self._load_us_landscape()
        }
        return landscape

    def _load_us_landscape(self) -> Dict:
        """
        Loads US political landscape data.
        """
        # Start with base knowledge (could be moved to a config or separate file)
        us_data = {
            "president": "Joe Biden",
            "party": "Democrat",
            "key_policies": [
                "Infrastructure Investment and Jobs Act",
                "Inflation Reduction Act",
                "CHIPS and Science Act",
                "Student Loan Relief efforts",
                "Support for Ukraine"
            ],
            "recent_developments": []
        }

        # Try to fetch recent developments
        try:
            recent_devs = self._fetch_recent_developments()
            if recent_devs:
                us_data["recent_developments"] = recent_devs
            else:
                 us_data["recent_developments"] = self.fallback_data["US"]["recent_developments"]
        except Exception as e:
            logger.error(f"Failed to fetch recent developments: {e}")
            us_data["recent_developments"] = self.fallback_data["US"]["recent_developments"]

        return us_data

    def _fetch_recent_developments(self) -> List[str]:
        """
        Fetches recent political developments from news sources.
        Returns a list of headlines.
        """
        developments = []

        # Method 1: Reuters US News (Scraping)
        # Note: Scraping is brittle. This is a best-effort implementation.
        try:
            developments.extend(self._scrape_reuters())
        except Exception as e:
            logger.warning(f"Reuters scraping failed: {e}")

        # Method 2: White House Briefing Room (Scraping)
        try:
            developments.extend(self._scrape_whitehouse())
        except Exception as e:
            logger.warning(f"White House scraping failed: {e}")

        return developments[:10] # Return top 10

    def _scrape_reuters(self) -> List[str]:
        """Scrapes headlines from Reuters."""
        url = self.sources["reuters_politics"]
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}

        try:
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code != 200:
                return []

            soup = BeautifulSoup(response.text, 'html.parser')
            headlines = []

            # Reuters structure changes often.
            # Look for h3 headers which usually contain article titles
            for h3 in soup.find_all('h3'):
                text = h3.get_text().strip()
                if len(text) > 20: # Filter out short navigations
                    headlines.append(f"[Reuters] {text}")

            return headlines
        except Exception as e:
            logger.error(f"Error scraping Reuters: {e}")
            return []

    def _scrape_whitehouse(self) -> List[str]:
        """Scrapes headlines from White House Briefing Room."""
        url = self.sources["whitehouse_briefing"]
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}

        try:
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code != 200:
                return []

            soup = BeautifulSoup(response.text, 'html.parser')
            headlines = []

            # Look for article titles in the briefing room list
            # Usually inside h2 with class 'news-item__title'
            for h2 in soup.find_all('h2', class_='news-item__title'):
                text = h2.get_text().strip()
                headlines.append(f"[White House] {text}")

            # Fallback if specific class not found
            if not headlines:
                 for h2 in soup.find_all('h2'):
                    a_tag = h2.find('a')
                    if a_tag:
                        text = a_tag.get_text().strip()
                        if len(text) > 20:
                            headlines.append(f"[White House] {text}")

            return headlines
        except Exception as e:
            logger.error(f"Error scraping White House: {e}")
            return []

if __name__ == "__main__":
    loader = PoliticalLandscapeLoader()
    print(loader.load_landscape())
