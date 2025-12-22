"""
Political Landscape Loader

This module provides functionality to load political landscape data from various sources.
It is designed to be used by the RegulatoryComplianceAgent and other reasoning engines.
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
    Supports 'Context Layering' for historical and dynamic geopolitical analysis.
    """

    def __init__(self):
        self.sources = {
            "whitehouse_briefing": "https://www.whitehouse.gov/briefing-room/",
            "reuters_politics": "https://www.reuters.com/world/us/",  # General US news/politics
        }
        # Fallback data in case of connection errors
        self.fallback_data = {
            "US": self._get_us_data_structure()
        }

    def load_landscape(self) -> Dict:
        """
        Loads the political landscape.
        Returns a rich object with current state, history, and cascading dynamics.
        """
        landscape = {
            "US": self._load_us_landscape()
        }
        return landscape

    def _get_us_data_structure(self) -> Dict:
        """
        Returns the structured data for the US political landscape.
        """
        return {
            # --- Layer 1: Current Administration (Surface Reality) ---
            "president": "Donald Trump",
            "party": "Republican",
            "term": "2025-2029",
            "key_policies": [
                "Tax Cuts and Jobs Act extensions",
                "Deregulation of Energy Sector",
                "Border Security and Immigration Enforcement",
                "Trade Tariffs (Universal Baseline Tariff)",
                "Energy Independence ('Drill, Baby, Drill')",
                "Dismantling of 'Deep State' bureaucracy"
            ],
            "recent_developments": [
                "Executive orders on energy production.",
                "Discussions on trade agreements and tariffs.",
                "Appointments of key cabinet positions."
            ],

            # --- Layer 2: Context Layering (Historical & Alternative) ---
            "context_layering": {
                "historical_context": {
                    "previous_administration": {
                        "president": "Joe Biden",
                        "party": "Democrat",
                        "term": "2021-2025",
                        "legacy_policies": [
                            "Infrastructure Investment and Jobs Act",
                            "Inflation Reduction Act",
                            "CHIPS and Science Act"
                        ]
                    },
                    "policy_shifts": {
                        "Energy": "Shift from Green Transition (IRA) to Fossil Fuel Maximization.",
                        "Trade": "Shift from Multilateral Frameworks to Bilateral Transactionalism.",
                        "Regulation": "Shift from ESG-focus to Deregulation."
                    }
                },
                "legislative_landscape": {
                    "senate_control": "Republican",
                    "house_control": "Republican",
                    "implication": "High probability of passing legislative agenda without gridlock."
                }
            },

            # --- Layer 3: Geopolitical Inter-Dynamics & Cascading Impacts ---
            # Used for Logic Reasoning and Critique Frameworks
            "geopolitical_dynamics": {
                "US_China_Decoupling": {
                    "trigger": "Universal Baseline Tariff & Revocation of PNTR",
                    "impact_ledger": {
                        "first_order": [
                            "Immediate increase in cost of goods sold (COGS) for electronics/retail.",
                            "Retaliatory tariffs on US agriculture (soybeans, corn)."
                        ],
                        "second_order": [
                            "Supply chain migration to Vietnam/India/Mexico (Nearshoring acceleration).",
                            "Inflationary pressure on US consumer durable goods (+2-4%).",
                            "Margin compression for US multinationals with high China revenue exposure."
                        ],
                        "third_order": [
                            "Bifurcation of global technology standards (6G, AI).",
                            "Potential selling of US Treasuries by China (Yield curve steepening).",
                            "Geopolitical flashpoints in Taiwan Strait due to economic pressure."
                        ]
                    }
                },
                "Energy_Dominance_Strategy": {
                    "trigger": "Unrestricted LNG Exports & Leasing on Federal Lands",
                    "impact_ledger": {
                        "first_order": [
                            "Decrease in global oil/gas prices due to US supply flood.",
                            "Increased revenue for US midstream and E&P companies."
                        ],
                        "second_order": [
                            "Fiscal strain on petrostates (Russia, Iran, Venezuela).",
                            "Reduced viability for green hydrogen/renewables projects without subsidies.",
                            "Lower input costs for US heavy industry (steel, chemicals)."
                        ],
                        "third_order": [
                            "Shift in EU energy security dependence from Russia/Middle East to US LNG.",
                            "Potential weakening of OPEC+ cohesion.",
                            "Long-term climate risk acceleration ( externalities not priced in)."
                        ]
                    }
                },
                "Regulatory_Deconstruction": {
                    "trigger": "Reinstatement of Schedule F & Agency Budget Cuts",
                    "impact_ledger": {
                        "first_order": [
                            "Immediate freeze on new federal regulations.",
                            "Reduction in compliance costs for financial/industrial sectors."
                        ],
                        "second_order": [
                            "Erosion of institutional memory and technical expertise in agencies.",
                            "Legal challenges from blue states (California effect).",
                            "Potential boost in short-term corporate profitability."
                        ],
                        "third_order": [
                            "Increased systemic risk due to weakened oversight (e.g., banking, aviation).",
                            "Fragmented regulatory landscape (State vs Federal conflict).",
                            "Privatization of previously public data/services."
                        ]
                    }
                }
            }
        }

    def _load_us_landscape(self) -> Dict:
        """
        Loads US political landscape data.
        """
        us_data = self._get_us_data_structure()

        # Try to fetch recent developments
        try:
            recent_devs = self._fetch_recent_developments()
            if recent_devs:
                us_data["recent_developments"] = recent_devs
        except Exception as e:
            logger.error(f"Failed to fetch recent developments: {e}")
            # Keep default/fallback developments
            pass

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

        return developments[:10]  # Return top 10

    def _scrape_reuters(self) -> List[str]:
        """Scrapes headlines from Reuters."""
        url = self.sources["reuters_politics"]
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}

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
                if len(text) > 20:  # Filter out short navigations
                    headlines.append(f"[Reuters] {text}")

            return headlines
        except Exception as e:
            logger.error(f"Error scraping Reuters: {e}")
            return []

    def _scrape_whitehouse(self) -> List[str]:
        """Scrapes headlines from White House Briefing Room."""
        url = self.sources["whitehouse_briefing"]
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}

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
