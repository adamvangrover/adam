# core/agents/event_driven_risk_agent.py

import datetime
import json
from typing import Dict, List

import requests
from utils.data_validation import validate_event_data
from utils.visualization_tools import generate_event_impact_chart

from .base_agent import BaseAgent


class EventDrivenRiskAgent(BaseAgent):
    """
    Agent that tracks and assesses the market impact of events.
    """

    def __init__(self, name: str = "EventDrivenRiskAgent"):
        super().__init__(name)
        self.event_data = {}  # Store event data
        self.api_key = "YOUR_NEWS_API_KEY" #replace with working api key
        self.news_api_url = "https://newsapi.org/v2/everything"

    def fetch_events(self, keywords: List[str], from_date: datetime.date, to_date: datetime.date) -> List[Dict]:
        """
        Fetches relevant news events from a news API.

        Args:
            keywords: List of keywords to search for.
            from_date: Start date for event search.
            to_date: End date for event search.

        Returns:
            List of event dictionaries.
        """
        events = []
        for keyword in keywords:
            params = {
                "q": keyword,
                "from": from_date.isoformat(),
                "to": to_date.isoformat(),
                "apiKey": self.api_key,
                "pageSize": 100, #max
            }
            try:
                response = requests.get(self.news_api_url, params=params, timeout=30)
                response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
                data = response.json()
                if data["status"] == "ok" and data["articles"]:
                    for article in data["articles"]:
                        event = {
                            "title": article["title"],
                            "description": article["description"],
                            "publishedAt": article["publishedAt"],
                            "url": article["url"],
                            "source": article["source"]["name"],
                            "keyword": keyword, # store the keyword used to find this article
                        }
                        if validate_event_data(event):
                            events.append(event)

            except requests.exceptions.RequestException as e:
                print(f"Error fetching news: {e}")
                return [] #return empty list on error
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
                return [] #return empty list on error
        return events

    def analyze_event_impact(self, events: List[Dict]) -> Dict:
        """
        Analyzes the impact of events on financial markets.

        Args:
            events: List of event dictionaries.

        Returns:
            Dictionary of event impact assessments.
        """
        impacts = {}
        for event in events:
            # Placeholder for impact analysis logic (e.g., sentiment analysis, historical correlation)
            # This is where more sophistacted analysis will be placed.
            impact_score = self.simulate_impact_analysis(event) #replace with actual analysis
            impacts[event["title"]] = {
                "impact_score": impact_score,
                "publishedAt": event["publishedAt"],
                "keyword": event["keyword"]
            }

        return impacts

    def generate_risk_alerts(self, event_impacts: Dict) -> List[Dict]:
        """
        Generates risk alerts based on event impact assessments.

        Args:
            event_impacts: Dictionary of event impact assessments.

        Returns:
            List of risk alert dictionaries.
        """
        alerts = []
        for event_title, impact_data in event_impacts.items():
            if impact_data["impact_score"] > 0.7:  # Threshold for high impact
                alert = {
                    "event": event_title,
                    "impact_score": impact_data["impact_score"],
                    "publishedAt": impact_data["publishedAt"],
                    "keyword": impact_data["keyword"],
                    "message": f"High impact event detected: {event_title} (Impact Score: {impact_data['impact_score']})"
                }
                alerts.append(alert)
        return alerts

    def simulate_impact_analysis(self, event: Dict) -> float:
        """
        Simulates impact analysis for testing purposes.
        Will be replaced with actual analysis.
        """
        # Placeholder for more sophisticated logic
        return hash(event["title"]) % 100 / 100.0 #returns a float between 0 and 1

    def generate_event_visualization(self, event_impacts:Dict):
        """Generates a visualization of event impacts"""
        generate_event_impact_chart(event_impacts)

    def run(self, keywords: List[str], from_date: datetime.date, to_date: datetime.date):
        """
        Runs the event-driven risk analysis.

        Args:
            keywords: List of keywords to search for.
            from_date: Start date for event search.
            to_date: End date for event search.
        """

        events = self.fetch_events(keywords, from_date, to_date)
        if not events: #handle error from fetch_events
          return
        impacts = self.analyze_event_impact(events)
        alerts = self.generate_risk_alerts(impacts)
        self.generate_event_visualization(impacts)
        return {"impacts": impacts, "alerts": alerts}

# Example usage (replace with actual dates and keywords)
if __name__ == "__main__":
    agent = EventDrivenRiskAgent()
    results = agent.run(keywords=["inflation", "interest rates"], from_date=datetime.date(2024, 1, 1), to_date=datetime.date(2024, 1, 31))
    print(json.dumps(results, indent=2))
