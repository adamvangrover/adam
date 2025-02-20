# core/agents/archive_manager_agent.py

import json

class ArchiveManagerAgent:
    def __init__(self, archive_dir):
        self.archive_dir = archive_dir

    def retrieve_market_overview(self, date):
        print(f"Retrieving market overview for {date}...")
        try:
            with open(f"{self.archive_dir}/market_overviews.json", "r") as f:
                market_overviews = json.load(f)
                for overview in market_overviews:
                    if overview["date"] == date:
                        return overview
                return None  # No overview found for that date
        except FileNotFoundError:
            return None
