#core/agents/supply_chain_risk_agent.py

import requests
import logging
from bs4 import BeautifulSoup
import folium
from geopy.geocoders import Nominatim

class SupplyChainRiskAgent:
    def __init__(self, news_api_key, supplier_data, transportation_routes, geopolitical_data, web_scraping_urls=None):
        self.news_api_key = news_api_key
        self.supplier_data = supplier_data  # List of suppliers with location information (latitude, longitude)
        self.transportation_routes = transportation_routes  # List of transportation routes with locations
        self.geopolitical_data = geopolitical_data  # Geopolitical factors affecting supply chains
        self.base_url = "https://newsapi.org/v2/everything"
        self.web_scraping_urls = web_scraping_urls or []
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)

    def fetch_news(self, query, language='en', page_size=5):
        """
        Fetches news articles related to the query (supply chain, geopolitical risks, etc.).
        :param query: Search query (keywords related to supply chain risks).
        :param language: Language of the articles (default: 'en').
        :param page_size: Number of articles to fetch per page (default: 5).
        :return: A list of articles.
        """
        params = {
            'q': query,
            'language': language,
            'pageSize': page_size,
            'apiKey': self.news_api_key
        }

        response = requests.get(self.base_url, params=params)
        if response.status_code == 200:
            return response.json()['articles']
        else:
            self.logger.error(f"Failed to fetch news. Status code: {response.status_code}")
            return []

    def fetch_web_scraped_data(self):
        """
        Fetches additional data through web scraping from the provided URLs.
        :return: A list of scraped articles or summaries.
        """
        scraped_data = []
        for url in self.web_scraping_urls:
            response = requests.get(url)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                paragraphs = soup.find_all('p')
                for p in paragraphs:
                    if p.text:
                        scraped_data.append(p.text.strip())
            else:
                self.logger.error(f"Failed to scrape {url}. Status code: {response.status_code}")
        return scraped_data

    def analyze_impact(self, articles):
        """
        Analyze the fetched articles for keywords and return a summary of the supply chain risks.
        :param articles: List of news articles.
        :return: A summary of the potential risks.
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
        :return: A map displaying high-risk locations.
        """
        geolocator = Nominatim(user_agent="supply_chain_risk_agent")
        risk_map = folium.Map(location=[20, 0], zoom_start=2)  # Start with a global view

        # Add supplier locations to map
        for supplier in self.supplier_data:
            location = geolocator.geocode(supplier['location'])
            if location:
                folium.Marker(
                    [location.latitude, location.longitude],
                    popup=f"Supplier: {supplier['name']}",
                    icon=folium.Icon(color='blue')
                ).add_to(risk_map)

        # Add transportation routes to map
        for route in self.transportation_routes:
            location = geolocator.geocode(route['location'])
            if location:
                folium.Marker(
                    [location.latitude, location.longitude],
                    popup=f"Transportation Route: {route['name']}",
                    icon=folium.Icon(color='green')
                ).add_to(risk_map)

        # Add geopolitical risk zones
        for risk_area in self.geopolitical_data:
            location = geolocator.geocode(risk_area['location'])
            if location:
                folium.Marker(
                    [location.latitude, location.longitude],
                    popup=f"Geopolitical Risk: {risk_area['risk_type']}",
                    icon=folium.Icon(color='red')
                ).add_to(risk_map)

        return risk_map

    def send_alert(self, risks):
        """
        Send alerts based on detected risks.
        :param risks: The analyzed risk data.
        """
        alert_level = 0
        for risk_type, articles in risks.items():
            if articles:
                alert_level += 1  # Increase alert level for each risk detected

        if alert_level >= 3:
            self.logger.warning("High supply chain risk detected! Immediate attention needed!")
        elif alert_level >= 1:
            self.logger.info("Moderate supply chain risk detected. Monitor closely.")
        else:
            self.logger.info("Supply chain risks are at a normal level.")

    def report_risks(self):
        """
        Fetches news articles and analyzes the risks, generating a report.
        :return: A report of the supply chain risks.
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
            self.logger.info("Web scraping provided additional information:")
            for item in scraped_data:
                self.logger.info(item)

        return all_risks

    def display_risk_report(self, risks):
        """
        Displays a summarized risk report.
        :param risks: The analyzed risk data.
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
    # Example of usage
    API_KEY = 'YOUR_NEWS_API_KEY'  # Replace with your NewsAPI key
    SUPPLIER_DATA = [
        {'name': 'Supplier A', 'location': 'New York, USA'},
        {'name': 'Supplier B', 'location': 'Shanghai, China'}
    ]
    TRANSPORTATION_ROUTES = [
        {'name': 'Route 1', 'location': 'Panama Canal'},
        {'name': 'Route 2', 'location': 'Suez Canal'}
    ]
    GEOPOLITICAL_DATA = [
        {'location': 'Ukraine', 'risk_type': 'Geopolitical Risk'},
        {'location': 'Taiwan', 'risk_type': 'Geopolitical Risk'}
    ]
    WEB_SCRAPING_URLS = [
        'https://www.example.com/supply-chain-news',  # Replace with actual URLs to scrape data
        'https://www.example.com/transportation-issues'
    ]
    
    # Instantiate the Supply Chain Risk Agent
    agent = SupplyChainRiskAgent(
        news_api_key=API_KEY,
        supplier_data=SUPPLIER_DATA,
        transportation_routes=TRANSPORTATION_ROUTES,
        geopolitical_data=GEOPOLITICAL_DATA,
        web_scraping_urls=WEB_SCRAPING_URLS
    )

    # Generate the risk report
    risks = agent.report_risks()
    agent.display_risk_report(risks)
    agent.send_alert(risks)

    # Generate and display risk map
    risk_map = agent.generate_risk_map()
    risk_map.save("supply_chain_risk_map.html")  # Save the map as an HTML file
    agent.logger.info("Risk map generated: supply_chain_risk_map.html")
