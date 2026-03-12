# core/agents/supply_chain_risk_agent.py

from __future__ import annotations
import requests
import logging
import os
import json
from typing import Dict, Any, List, Union
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import ipaddress
import asyncio

try:
    import folium
    FOLIUM_AVAILABLE = True
except ImportError:
    FOLIUM_AVAILABLE = False

try:
    from geopy.geocoders import Nominatim
    GEOPY_AVAILABLE = True
except ImportError:
    GEOPY_AVAILABLE = False

from core.agents.agent_base import AgentBase
from core.schemas.agent_schema import AgentInput, AgentOutput

logger = logging.getLogger(__name__)


class SupplyChainRiskAgent(AgentBase):
    """
    Agent responsible for assessing risks in global supply chains.
    """

    def __init__(self, config: Dict[str, Any], **kwargs):
        super().__init__(config, **kwargs)
        self.news_api_key = self.config.get('news_api_key') or os.environ.get('NEWS_API_KEY')
        self.supplier_data = self.config.get('supplier_data', [])
        self.transportation_routes = self.config.get('transportation_routes', [])
        self.geopolitical_data = self.config.get('geopolitical_data', [])
        self.web_scraping_urls = self.config.get('web_scraping_urls', [])
        self.base_url = "https://newsapi.org/v2/everything"

    async def execute(self, input_data: Union[str, AgentInput, Dict[str, Any]] = None, **kwargs) -> Union[Dict[str, Any], AgentOutput]:
        """
        Executes supply chain risk analysis.
        """
        logger.info("Executing SupplyChainRiskAgent...")

        is_standard_mode = False
        query = "Supply Chain Risk Report"

        if input_data is not None:
            if isinstance(input_data, AgentInput):
                query = input_data.query
                is_standard_mode = True

                # Context overrides
                context = input_data.context
                if "supplier_data" in context: self.supplier_data = context["supplier_data"]
                if "transportation_routes" in context: self.transportation_routes = context["transportation_routes"]
                if "geopolitical_data" in context: self.geopolitical_data = context["geopolitical_data"]
            elif isinstance(input_data, dict):
                kwargs.update(input_data)
            elif isinstance(input_data, str):
                query = input_data

        # Fallbacks to kwargs
        if "supplier_data" in kwargs: self.supplier_data = kwargs["supplier_data"]

        # Run Analysis
        risks = self.report_risks(query)

        # Try to generate map
        map_generated = False
        if FOLIUM_AVAILABLE and GEOPY_AVAILABLE and self.supplier_data:
            try:
                risk_map = self.generate_risk_map()
                if risk_map:
                    os.makedirs("downloads", exist_ok=True)
                    risk_map.save("downloads/supply_chain_risk_map.html")
                    map_generated = True
            except Exception as e:
                logger.warning(f"Failed to generate map: {e}")

        # Send alert if high risk
        alert_level = self.send_alert(risks)

        result = {
            "query": query,
            "risks_detected": risks,
            "alert_level": alert_level,
            "map_generated": map_generated,
            "suppliers_monitored": len(self.supplier_data)
        }

        if is_standard_mode:
            answer = f"Supply Chain Risk Report for '{query}':\n\n"
            answer += f"Alert Level: {alert_level}/3\n"

            for risk_type, items in risks.items():
                if items:
                    answer += f"\n- {risk_type.replace('_', ' ').title()} ({len(items)} alerts):\n"
                    for item in items[:3]: # Show top 3
                        answer += f"  * {item}\n"

            if map_generated:
                answer += "\nNote: Interactive risk map generated at 'downloads/supply_chain_risk_map.html'."

            return AgentOutput(
                answer=answer,
                sources=["NewsAPI", "Internal Routing Data"],
                confidence=0.8 if self.news_api_key else 0.4,
                metadata=result
            )

        return result

    def fetch_news(self, query, language='en', page_size=5) -> List[Dict]:
        """
        Fetches news articles. Mocks if API key is missing.
        """
        if not self.news_api_key or self.news_api_key.startswith("YOUR_"):
            logger.debug("No valid NEWS_API_KEY. Using mock news data.")
            if "disruption" in query:
                return [{"title": "Port strike causes major backlog", "description": "Delays expected for weeks."}]
            elif "geopolitical" in query:
                return [{"title": "New tariffs imposed on critical minerals", "description": "Supply chain restructuring anticipated."}]
            return []

        params = {
            'q': query,
            'language': language,
            'pageSize': page_size,
            'apiKey': self.news_api_key
        }

        try:
            response = requests.get(self.base_url, params=params, timeout=10)
            if response.status_code == 200:
                return response.json().get('articles', [])
            else:
                logger.warning(f"Failed to fetch news. Status code: {response.status_code}")
                return []
        except requests.RequestException as e:
            logger.error(f"Request failed: {e}")
            return []

    def _is_safe_url(self, url: str) -> bool:
        """SSRF Protection."""
        try:
            parsed = urlparse(url)
            if parsed.scheme not in ('http', 'https'):
                return False

            hostname = parsed.hostname
            if not hostname:
                return False

            try:
                ip_obj = ipaddress.ip_address(hostname)
                if ip_obj.is_private or ip_obj.is_loopback:
                    return False
            except ValueError:
                if hostname in ('localhost', '127.0.0.1', '::1'):
                    return False
            return True
        except Exception:
            return False

    def fetch_web_scraped_data(self) -> List[str]:
        scraped_data = []
        for url in self.web_scraping_urls:
            if not self._is_safe_url(url):
                continue
            try:
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.content, 'html.parser')
                    for p in soup.find_all('p')[:2]: # Limit to first 2 paragraphs
                        if p.text:
                            scraped_data.append(p.text.strip())
            except requests.RequestException:
                pass
        return scraped_data

    def analyze_impact(self, articles: List[Dict]) -> Dict[str, List[str]]:
        risk_summary = {'disruptions': [], 'geopolitical_risks': [], 'transportation_risks': []}

        for article in articles:
            title = article.get('title', '').lower()
            description = article.get('description', '').lower()
            content = article.get('content', '').lower()

            for keyword in ['supply chain disruption', 'natural disaster', 'geopolitical risk', 'trade policy', 'strike', 'tariff']:
                if keyword in title or keyword in description or keyword in content:
                    if 'disruption' in title or 'strike' in title:
                        risk_summary['disruptions'].append(article['title'])
                    elif 'geopolitical' in title or 'trade' in title or 'tariff' in title:
                        risk_summary['geopolitical_risks'].append(article['title'])
                    elif 'transportation' in title or 'port' in title:
                        risk_summary['transportation_risks'].append(article['title'])

        return risk_summary

    def generate_risk_map(self):
        if not FOLIUM_AVAILABLE or not GEOPY_AVAILABLE:
            logger.warning("Folium or Geopy not installed. Cannot generate map.")
            return None

        geolocator = Nominatim(user_agent="adam_sc_agent")
        risk_map = folium.Map(location=[20, 0], zoom_start=2)

        for supplier in self.supplier_data:
            try:
                location = geolocator.geocode(supplier['location'])
                if location:
                    folium.Marker([location.latitude, location.longitude], popup=f"Supplier: {supplier['name']}", icon=folium.Icon(color='blue')).add_to(risk_map)
            except Exception:
                pass # Ignore geocoding errors

        return risk_map

    def send_alert(self, risks: Dict[str, List[str]]) -> int:
        alert_level = sum(1 for items in risks.values() if items)
        if alert_level >= 3:
            logger.warning("High supply chain risk detected! Immediate attention needed!")
        elif alert_level >= 1:
            logger.info("Moderate supply chain risk detected. Monitor closely.")
        else:
            logger.info("Supply chain risks are at a normal level.")
        return alert_level

    def report_risks(self, custom_query: str = None) -> Dict[str, List[str]]:
        all_risks = {'disruptions': [], 'geopolitical_risks': [], 'transportation_risks': []}

        queries = ['supply chain disruption', 'geopolitical risk', 'transportation issues']
        if custom_query and custom_query not in queries:
             queries.append(custom_query)

        for keyword in queries:
            articles = self.fetch_news(query=keyword)
            if articles:
                risk_analysis = self.analyze_impact(articles)
                for risk_type in risk_analysis:
                    all_risks[risk_type].extend(risk_analysis[risk_type])

        # Deduplicate
        for k in all_risks:
            all_risks[k] = list(set(all_risks[k]))

        scraped_data = self.fetch_web_scraped_data()
        if scraped_data:
            all_risks['disruptions'].extend(scraped_data)

        return all_risks
