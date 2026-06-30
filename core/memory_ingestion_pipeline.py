import os
import re
from typing import Dict, List, Optional
from dataclasses import dataclass, field

@dataclass
class MemoryNode:
    date_id: str
    equities: Dict[str, str] = field(default_factory=dict)
    fixed_income: Dict[str, str] = field(default_factory=dict)
    commodities: Dict[str, str] = field(default_factory=dict)
    narrative_summary: str = ""

class MemoryIngestionPipeline:
    """
    Ingests Markdown memory nodes and structures them into key-value
    pairs suitable for Vector DB ingestion and dynamic prompt injection.
    """
    def __init__(self, memory_dir: str = "artifacts/ai/"):
        self.memory_dir = memory_dir
        
    def _parse_markdown_section(self, section_text: str) -> Dict[str, str]:
        """Extracts bulleted items into a key-value dictionary."""
        data = {}
        # Matches patterns like "* **S&P 500:** 7,321.15"
        pattern = re.compile(r'\*\s*\*\*(.*?):\*\*\s*(.*?)(?:\(Target|$)')
        for line in section_text.split('\n'):
            match = pattern.search(line)
            if match:
                key = match.group(1).strip()
                value = match.group(2).strip()
                data[key] = value
        return data

    def ingest_node(self, filename: str) -> MemoryNode:
        """Parses a specific Markdown memory node."""
        filepath = os.path.join(self.memory_dir, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Memory node {filepath} not found.")
            
        with open(filepath, 'r') as f:
            content = f.read()

        # Extract Date ID from filename (e.g., Market_State_20260629.md)
        date_id_match = re.search(r'(\d{8})', filename)
        date_id = date_id_match.group(1) if date_id_match else "Unknown"

        node = MemoryNode(date_id=date_id)

        # Parse Equities
        equities_match = re.search(r'\*\*(?:Equities).*?\*\*(.*?)(?:\n\n|\Z)', content, re.DOTALL)
        if equities_match:
            node.equities = self._parse_markdown_section(equities_match.group(1))

        # Parse Fixed Income
        fixed_match = re.search(r'\*\*(?:Fixed Income).*?\*\*(.*?)(?:\n\n|\Z)', content, re.DOTALL)
        if fixed_match:
            node.fixed_income = self._parse_markdown_section(fixed_match.group(1))

        # Parse Commodities
        comm_match = re.search(r'\*\*(?:Commodities).*?\*\*(.*?)(?:\n\n|\Z)', content, re.DOTALL)
        if comm_match:
            node.commodities = self._parse_markdown_section(comm_match.group(1))

        return node

    def format_for_vector_db(self, node: MemoryNode) -> List[Dict[str, str]]:
        """Converts the structured node into key-value pairs for vector indexing."""
        documents = []
        
        def _create_doc(category: str, k: str, v: str):
            return {
                "id": f"{node.date_id}_{category}_{k.replace(' ', '_')}",
                "text": f"On {node.date_id}, the {k} value was {v}.",
                "metadata": {"date": node.date_id, "category": category, "asset": k}
            }

        for k, v in node.equities.items():
            documents.append(_create_doc("Equity", k, v))
            
        for k, v in node.fixed_income.items():
            documents.append(_create_doc("Fixed_Income", k, v))
            
        for k, v in node.commodities.items():
            documents.append(_create_doc("Commodity", k, v))

        return documents

    def generate_conversational_summary(self, node: MemoryNode, delta_node: Optional[MemoryNode] = None) -> str:
        """
        Generates a conversational summary for dynamic System Prompt injection.
        If a delta_node (previous day) is provided, it calculates the delta.
        """
        summary = f"System Context (Date: {node.date_id}):\n"
        summary += "The current market state is characterized by "
        
        if "S&P 500" in node.equities:
            summary += f"the S&P 500 trading at {node.equities['S&P 500']}. "
            
        if "US 10Y Treasury Yield" in node.fixed_income:
            summary += f"Bond markets see the 10Y Yield at {node.fixed_income['US 10Y Treasury Yield']}, "
            
        if "US High Yield OAS" in node.fixed_income:
            summary += f"with High Yield OAS spreads at {node.fixed_income['US High Yield OAS']}. "
            
        if "Gold (Spot)" in node.commodities:
            summary += f"In commodities, Gold is priced at {node.commodities['Gold (Spot)']}. "

        if delta_node:
            summary += "\n\nDelta Analysis:\n"
            # Simple delta example for S&P 500
            try:
                curr_sp = float(node.equities.get('S&P 500', '0').replace(',', ''))
                prev_sp = float(delta_node.equities.get('S&P 500', '0').replace(',', ''))
                diff = curr_sp - prev_sp
                direction = "higher" if diff >= 0 else "lower"
                summary += f"The S&P 500 is {abs(diff):.2f} points {direction} than the previous session."
            except ValueError:
                pass
                
        return summary

# Example Usage
if __name__ == "__main__":
    pipeline = MemoryIngestionPipeline()
    # In a real run, you'd iterate through files in artifacts/ai/
    print("Memory Ingestion Pipeline initialized.")
