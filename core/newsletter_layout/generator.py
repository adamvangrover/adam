
import os
import json
import glob
from datetime import datetime
from jinja2 import Environment, FileSystemLoader
from typing import Dict, Any, List

from core.utils.logging_utils import get_logger

logger = get_logger(__name__)

TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), 'templates')


class NewsletterGenerator:
    """
    Adaptive Newsletter Generator using Jinja2 templates.
    """

    def __init__(self, template_dir: str = TEMPLATE_DIR):
        self.env = Environment(loader=FileSystemLoader(template_dir), autoescape=True)
        self.snapshot_dir = "data/snapshots"

    def load_latest_snapshot(self) -> Dict[str, Any]:
        """Loads the most recent UKG snapshot."""
        list_of_files = glob.glob(os.path.join(self.snapshot_dir, "*.json"))
        if not list_of_files:
            logger.warning("No snapshots found.")
            return {}

        latest_file = max(list_of_files, key=os.path.getctime)
        logger.info(f"Loading snapshot: {latest_file}")

        with open(latest_file, 'r') as f:
            return json.load(f)

    def extract_graph_data(self, graph_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transforms flat graph data into structured objects for templates.
        """
        nodes = graph_data.get("nodes", [])

        # Maps
        companies = {}
        snapshots = []
        news_items = []
        indices = []

        # Index nodes by ID for easier linking if needed
        node_map = {n['id']: n for n in nodes} if nodes and 'id' in nodes[0] else {}

        for node in nodes:
            # Handle NetworkX node_link_data format which usually puts ID in 'id'
            attrs = node
            node_type = attrs.get("type")

            if node_type == "Company":
                companies[attrs.get("id")] = attrs
            elif node_type == "MarketSnapshot":
                snapshots.append(attrs)
            elif node_type == "MarketIndex":
                indices.append(attrs)
            elif node_type == "NewsArticle":
                # Find associated ticker?
                # We need edges to link News -> Company.
                # Since we don't have edges easily accessible in this simple loop without building a graph,
                # we can rely on the ID naming convention "News:TICKER:IDX" used in ingestion.
                node_id = attrs.get("id", "")
                parts = node_id.split(":")
                if len(parts) >= 2 and parts[0] == "News":
                    attrs['ticker'] = parts[1]

                news_items.append(attrs)

        # Sort snapshots by ticker
        snapshots.sort(key=lambda x: x.get("ticker", ""))

        # Group news by ticker
        news_by_ticker = {}
        for item in news_items:
            t = item.get("ticker")
            if t:
                if t not in news_by_ticker:
                    news_by_ticker[t] = []
                news_by_ticker[t].append(item)

        return {
            "companies": companies,
            "snapshots": snapshots,
            "news": news_by_ticker,
            "indices": indices,
            "date": datetime.now().strftime("%B %d, %Y"),
            "timestamp": datetime.now().isoformat()
        }

    def generate(self, template_name: str, output_path: str = None) -> str:
        """
        Generates a newsletter using the specified template.
        """
        try:
            template = self.env.get_template(template_name)

            raw_data = self.load_latest_snapshot()
            if not raw_data:
                return "Error: No data available."

            context = self.extract_graph_data(raw_data)

            rendered_content = template.render(**context)

            if output_path:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                with open(output_path, 'w') as f:
                    f.write(rendered_content)
                logger.info(f"Newsletter generated at {output_path}")

            return rendered_content

        except Exception as e:
            logger.error(f"Failed to generate newsletter: {e}")
            raise
