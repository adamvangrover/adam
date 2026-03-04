# core/agents/supply_chain_risk_agent.py

import requests
import logging
import asyncio
from bs4 import BeautifulSoup
import folium
from geopy.geocoders import Nominatim
from urllib.parse import urlparse
import ipaddress
from typing import Dict, Any, List, Optional
import os

from core.agents.agent_base import AgentBase

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SupplyChainRiskAgent(AgentBase):
    """
    Agent responsible for assessing supply chain risks using news and scraping.
    """

    def __init__(self, config: Dict[str, Any], kernel: Optional[Any] = None):
        """
        Initializes the SupplyChainRiskAgent.

        Args:
            config (dict): Configuration dictionary.
            kernel (Optional[Any]): Semantic Kernel instance.
        """
        super().__init__(config, kernel=kernel)
        self.news_api_key = self.config.get('news_api_key') or os.environ.get('NEWS_API_KEY')
        self.supplier_data = self.config.get('supplier_data', [])
        self.transportation_routes = self.config.get('transportation_routes', [])
        self.geopolitical_data = self.config.get('geopolitical_data', [])
        self.web_scraping_urls = self.config.get('web_scraping_urls', [])
        self.base_url = "https://newsapi.org/v2/everything"
        self.logger = logging.getLogger(__name__)

        if not self.news_api_key:
            self.logger.warning("NEWS_API_KEY is not set. News fetching will fail.")

    async def execute(self, *args, **kwargs):
        """
        Executes the supply chain risk assessment.

        Returns:
            dict: A report of supply chain risks.
        """
        self.logger.info("Starting supply chain risk assessment...")

        # Optionally update data from kwargs if provided
        if 'supplier_data' in kwargs: self.supplier_data = kwargs['supplier_data']
        if 'transportation_routes' in kwargs: self.transportation_routes = kwargs['transportation_routes']

        loop = asyncio.get_running_loop()

        # Run synchronous network operations in a thread pool
        risks = await loop.run_in_executor(None, self.report_risks)

        # Generate map if requested
        if kwargs.get('generate_map', False):
             risk_map = await loop.run_in_executor(None, self.generate_risk_map)
             # Saving map logic could be moved here or kept in method
             # For now, just return status
             map_path = "supply_chain_risk_map.html"
             risk_map.save(map_path)
             risks['map_generated'] = map_path

        self.display_risk_report(risks)
        self.send_alert(risks)

        return risks

    def fetch_news(self, query, language='en', page_size=5):
        """
        Fetches news articles related to the query (supply chain, geopolitical risks, etc.).
        """
        if not self.news_api_key:
            return []

        params = {
            'q': query,
            'language': language,
            'pageSize': page_size,
            'apiKey': self.news_api_key
        }

        # 🛡️ Sentinel: Added timeout to prevent DoS via slow connections
        try:
            response = requests.get(self.base_url, params=params, timeout=10)
            if response.status_code == 200:
                return response.json().get('articles', [])
            else:
                self.logger.error(f"Failed to fetch news. Status code: {response.status_code}")
                return []
        except requests.RequestException as e:
            self.logger.error(f"Request failed: {e}")
            return []

    def _is_safe_url(self, url):
        """
        Validates the URL to prevent SSRF attacks.
        """
        try:
            parsed = urlparse(url)
            if parsed.scheme not in ('http', 'https'):
                self.logger.warning(f"Blocked unsafe scheme: {parsed.scheme}")
                return False

            hostname = parsed.hostname
            if not hostname:
                return False

            # Check if hostname is an IP address
            try:
                ip_obj = ipaddress.ip_address(hostname)
                if ip_obj.is_private or ip_obj.is_loopback:
                    self.logger.warning(f"Blocked private IP: {hostname}")
                    return False
            except ValueError:
                # Hostname check
                if hostname in ('localhost', '127.0.0.1', '::1'):
                    self.logger.warning(f"Blocked localhost: {hostname}")
                    return False

            return True

        except Exception as e:
            self.logger.error(f"URL validation failed for {url}: {e}")
            return False

    def fetch_web_scraped_data(self):
        """
        Fetches additional data through web scraping from the provided URLs.
        """
        scraped_data = []
        for url in self.web_scraping_urls:
            # 🛡️ Sentinel: SSRF Protection
            if not self._is_safe_url(url):
                self.logger.warning(f"Skipping unsafe URL: {url}")
                continue

            try:
                # Add timeout to prevent DoS via slow connections
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.content, 'html.parser')
                    paragraphs = soup.find_all('p')
                    for p in paragraphs:
                        if p.text:
                            scraped_data.append(p.text.strip())
                else:
                    self.logger.error(f"Failed to scrape {url}. Status code: {response.status_code}")
            except requests.RequestException as e:
                self.logger.error(f"Request failed for {url}: {e}")

        return scraped_data

    def analyze_impact(self, articles):
        """
        Analyze the fetched articles for keywords and return a summary of the supply chain risks.
        """
        risk_summary = {'disruptions': [], 'geopolitical_risks': [], 'transportation_risks': []}

        for article in articles:
            title = article.get('title', '').lower()
            description = article.get('description', '').lower()
            content = article.get('content', '').lower()

            for keyword in ['supply chain disruption', 'natural disaster', 'geopolitical risk', 'trade policy']:
                if keyword in title or keyword in description or keyword in content:
                    if 'disruption' in title or 'disruption' in description:
                        risk_summary['disruptions'].append(article['title'])
                    elif 'geopolitical' in title or 'trade policy' in description:
                        risk_summary['geopolitical_risks'].append(article['title'])
                    elif 'transportation' in title or 'transport' in description:
                        risk_summary['transportation_risks'].append(article['title'])

        return risk_summary

    def generate_risk_map(self):
        """
        Generate a simple risk map to visualize supplier and transportation risk locations.
        """
        geolocator = Nominatim(user_agent="supply_chain_risk_agent")
        risk_map = folium.Map(location=[20, 0], zoom_start=2)  # Start with a global view

        # Add supplier locations to map
        for supplier in self.supplier_data:
            try:
                location = geolocator.geocode(supplier['location'])
                if location:
                    folium.Marker(
                        [location.latitude, location.longitude],
                        popup=f"Supplier: {supplier['name']}",
                        icon=folium.Icon(color='blue')
                    ).add_to(risk_map)
            except Exception as e:
                self.logger.error(f"Geocoding failed for supplier {supplier}: {e}")

        # Add transportation routes to map
        for route in self.transportation_routes:
            try:
                location = geolocator.geocode(route['location'])
                if location:
                    folium.Marker(
                        [location.latitude, location.longitude],
                        popup=f"Transportation Route: {route['name']}",
                        icon=folium.Icon(color='green')
                    ).add_to(risk_map)
            except Exception as e:
                self.logger.error(f"Geocoding failed for route {route}: {e}")

        # Add geopolitical risk zones
        for risk_area in self.geopolitical_data:
            try:
                location = geolocator.geocode(risk_area['location'])
                if location:
                    folium.Marker(
                        [location.latitude, location.longitude],
                        popup=f"Geopolitical Risk: {risk_area['risk_type']}",
                        icon=folium.Icon(color='red')
                    ).add_to(risk_map)
            except Exception as e:
                self.logger.error(f"Geocoding failed for risk area {risk_area}: {e}")

        return risk_map

    def send_alert(self, risks):
        """
        Send alerts based on detected risks.
        """
        alert_level = 0
        for risk_type, articles in risks.items():
            if articles:
                alert_level += 1

        if alert_level >= 3:
            self.logger.warning("High supply chain risk detected! Immediate attention needed!")
        elif alert_level >= 1:
            self.logger.info("Moderate supply chain risk detected. Monitor closely.")
        else:
            self.logger.info("Supply chain risks are at a normal level.")

    def report_risks(self):
        """
        Fetches news articles and analyzes the risks, generating a report.
        """
        all_risks = {'disruptions': [], 'geopolitical_risks': [], 'transportation_risks': []}

        # Fetch and analyze news articles
        for keyword in ['supply chain disruption', 'geopolitical risk', 'transportation issues']:
            self.logger.info(f"Fetching news for keyword: {keyword}")
            articles = self.fetch_news(query=keyword)

            if articles:
                risk_analysis = self.analyze_impact(articles)
                for risk_type in risk_analysis:
                    all_risks[risk_type].extend(risk_analysis[risk_type])

        # Fetch additional data via web scraping
        scraped_data = self.fetch_web_scraped_data()
        if scraped_data:
            self.logger.info("Web scraping provided additional information (summary):")
            self.logger.info(f"Scraped {len(scraped_data)} items.")
            # Optionally analyze scraped data here

        return all_risks

    def display_risk_report(self, risks):
        """
        Displays a summarized risk report.
        """
        self.logger.info("\nSupply Chain Risk Report:")
        for risk_type, articles in risks.items():
            if articles:
                self.logger.info(f"\n{risk_type.replace('_', ' ').title()}:")
                for article in articles:
                    self.logger.info(f"- {article}")
            else:
                self.logger.info(f"\nNo {risk_type.replace('_', ' ')} detected.")


if __name__ == "__main__":
    import asyncio

    # Example Config
    config = {
        'news_api_key': os.environ.get('NEWS_API_KEY', 'mock_key'),
        'supplier_data': [{'name': 'Supplier A', 'location': 'New York, USA'}],
        'transportation_routes': [{'name': 'Route 1', 'location': 'Panama Canal'}],
        'geopolitical_data': [{'location': 'Ukraine', 'risk_type': 'Geopolitical Risk'}]
    }

    agent = SupplyChainRiskAgent(config)

    async def main():
        await agent.execute()

    asyncio.run(main())
