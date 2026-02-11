#!/usr/bin/env python3
"""
MarketMayhem Bridge Script
--------------------------
Connects the MarketMayhem generator (simulated/newsletter layout specialist)
to the House View and Strategy HTML dashboards for real-time system views.
"""

import json
import re
import random
import datetime
from pathlib import Path

# Placeholder for actual NewsletterLayoutSpecialist if needed
# from core.agents.newsletter_layout_specialist_agent import NewsletterLayoutSpecialist

class MarketMayhemBridge:
    def __init__(self):
        self.scenarios = [
            "Flash Crash: HFT Algorithms go rogue.",
            "Liquidity Crunch: Bond yields spike unexpectedly.",
            "AI Bubble Burst: Tech sector valuation correction.",
            "Crypto Winter: Regulatory crackdown on digital assets.",
            "Currency War: Major forex pairs diverge wildly.",
            "Energy Crisis: Oil supply shock due to geopolitical tension."
        ]

    def generate_chaos_data(self):
        """
        Simulates the generation of Market Mayhem data.
        Returns a dictionary with system metrics and a narrative summary.
        """
        scenario = random.choice(self.scenarios)
        volatility = round(random.uniform(15.0, 45.0), 2)
        liquidity_stress = round(random.uniform(0.0, 100.0), 1)
        system_load = round(random.uniform(20.0, 95.0), 1)
        active_agents = random.randint(5, 50)

        # Simulated metrics for charts
        metrics = {
            "System Volatility (VIX)": volatility,
            "Liquidity Stress Index": liquidity_stress,
            "System Load (%)": system_load,
            "Active Agents": active_agents,
            "Scenario Impact Score": round(random.uniform(1.0, 10.0), 1)
        }

        # Simulated narrative
        narrative = f"""
        <h2>Executive Summary (LIVE)</h2>
        <ul>
        <li><strong>Scenario:</strong> {scenario}</li>
        <li><strong>Alert:</strong> System volatility is at {volatility}. Immediate attention required.</li>
        <li><strong>Liquidity:</strong> Stress levels at {liquidity_stress}/100. Monitoring banking sector.</li>
        <li><strong>Agent Activity:</strong> {active_agents} autonomous agents deployed to mitigate risks.</li>
        </ul>
        """

        return {
            "metrics": metrics,
            "narrative": narrative,
            "scenario": scenario,
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

    def inject_data(self, template_path, output_path, data):
        """
        Injects the generated data into the HTML template.
        """
        try:
            with open(template_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # 1. Inject Metrics into JS
            # For house_view_20251210.html: window.REPORT_METRICS = {};
            metrics_json = json.dumps(data['metrics'], indent=4)
            if "window.REPORT_METRICS = {};" in content:
                content = content.replace(
                    "window.REPORT_METRICS = {};",
                    f"window.REPORT_METRICS = {metrics_json};"
                )

            # For market_pulse_20250315.html: const metricsData = {};
            if "const metricsData = {};" in content:
                content = content.replace(
                    "const metricsData = {};",
                    f"const metricsData = {metrics_json};"
                )

            # 2. Inject Narrative (Executive Summary)
            # Find the existing Executive Summary block and replace it
            # This regex looks for <h2>Executive Summary</h2> followed by a <ul> list
            # We use non-greedy matching for the content inside <ul>
            narrative_pattern = re.compile(r'<h2>Executive Summary</h2>\s*<ul>.*?</ul>', re.DOTALL)
            if narrative_pattern.search(content):
                content = narrative_pattern.sub(data['narrative'], content)
            else:
                # Fallback if structure is different or not found, try appending or replacing a placeholder
                print(f"Warning: Could not find 'Executive Summary' block in {template_path}")

            # 3. Update Metadata/Title to indicate LIVE
            content = content.replace("<title>", "<title>[LIVE] ")

            # Update H1 tag which likely has attributes (e.g., class="title")
            h1_pattern = re.compile(r'(<h1[^>]*>)')
            if h1_pattern.search(content):
                content = h1_pattern.sub(r'\1[LIVE] ', content)
            else:
                # Fallback for simple h1
                content = content.replace("<h1>", "<h1>[LIVE] ")

            # Update specific date/timestamp placeholders if possible
            # Replacing YYYY-MM-DD pattern might be risky if multiple exist, but let's try a specific one if known
            # content = re.sub(r'\d{4}-\d{2}-\d{2}', data['timestamp'].split(' ')[0], content)

            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(content)

            print(f"Successfully generated {output_path}")
            return True

        except Exception as e:
            print(f"Error processing {template_path}: {e}")
            return False

    def run(self):
        print("Starting MarketMayhem Bridge...")
        data = self.generate_chaos_data()
        print(f"Generated Scenario: {data['scenario']}")
        print(f"Metrics: {data['metrics']}")

        # Paths
        base_dir = Path("showcase")

        # Target 1: House View
        template_hv = base_dir / "house_view_20251210.html"
        output_hv = base_dir / "live_house_view.html"
        if template_hv.exists():
            self.inject_data(template_hv, output_hv, data)
        else:
            print(f"Template not found: {template_hv}")

        # Target 2: Market Pulse
        template_mp = base_dir / "market_pulse_20250315.html"
        output_mp = base_dir / "live_market_pulse.html"
        if template_mp.exists():
            self.inject_data(template_mp, output_mp, data)
        else:
            print(f"Template not found: {template_mp}")

        print("Bridge execution complete.")

if __name__ == "__main__":
    bridge = MarketMayhemBridge()
    bridge.run()
