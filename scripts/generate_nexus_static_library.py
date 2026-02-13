#!/usr/bin/env python3
import json
import os
import random

# Configuration
NEXUS_DATA_PATH = "showcase/data/nexus_simulation.json"
OUTPUT_PATH = "showcase/data/nexus_static_library.json"

def main():
    print("Generating Expanded Nexus Static Library...")

    # 1. Load Sovereigns
    sovereigns = []
    if os.path.exists(NEXUS_DATA_PATH):
        try:
            with open(NEXUS_DATA_PATH, 'r') as f:
                data = json.load(f)
                nodes = data.get("graph", {}).get("nodes", [])
                sovereigns = [n.get("label") for n in nodes if n.get("label")]
        except Exception as e:
            print(f"Error loading nexus_simulation.json: {e}")
            return
    else:
        sovereigns = ["NorthAm-Market-1", "EuroZone-Digital-2", "AsiaPac-Resource-3"]

    # Add Shadow Factions (Hidden nodes for simulation depth)
    shadow_factions = [
        "The Syndicate", "Apex Collective", "Node Zero", "Iron Bank Protocol",
        "Cipher Vanguard", "Echo Cell", "Obsidian Group"
    ]

    print(f"Found {len(sovereigns)} sovereigns + {len(shadow_factions)} shadow factions.")

    # 2. Expanded Event Ontology with Metrics
    # Impact ranges: Stability (-1.0 to 1.0), Sentiment (0 to 100), Conviction (0 to 100)
    event_templates = [
        # --- 1st Order: Geopolitical / Macro ---
        {"type": "Cyber Operation", "impact": -0.05, "sentiment": -5, "conviction": -2, "desc": "Targeting critical infrastructure"},
        {"type": "Military Maneuver", "impact": -0.1, "sentiment": -15, "conviction": 5, "desc": "Mobilizing assets near contested zone"},
        {"type": "Trade Sanction", "impact": -0.02, "sentiment": -10, "conviction": 2, "desc": "Imposing tariffs on key exports"},
        {"type": "Diplomatic Summit", "impact": 0.05, "sentiment": 10, "conviction": -5, "desc": "Hosting peace talks"},
        {"type": "Strategic Alliance", "impact": 0.03, "sentiment": 15, "conviction": 10, "desc": "Signing mutual defense treaty"},
        {"type": "Resource Discovery", "impact": 0.08, "sentiment": 20, "conviction": 5, "desc": "Massive lithium deposit found"},

        # --- 1st Order: Economic ---
        {"type": "Currency Devaluation", "impact": -0.04, "sentiment": -10, "conviction": -10, "desc": "Central bank intervenes to stabilize currency"},
        {"type": "Liquidity Crisis", "impact": -0.07, "sentiment": -20, "conviction": -15, "desc": "Interbank lending freezes unexpectedly"},
        {"type": "Market Rally", "impact": 0.04, "sentiment": 15, "conviction": 10, "desc": "Tech sector leads global breakout"},

        # --- 1st Order: Tech / Anomaly ---
        {"type": "AI Hallucination", "impact": -0.02, "sentiment": -5, "conviction": -20, "desc": "Automated defense systems trigger false alarm"},
        {"type": "Satellite Outage", "impact": -0.04, "sentiment": -8, "conviction": -5, "desc": "Global GPS network disruption reported"},
        {"type": "Quantum Breakthrough", "impact": 0.1, "sentiment": 25, "conviction": 20, "desc": "Encryption standards rendered obsolete"},

        # --- 2nd Order: Consequences (Triggered by 1st) ---
        {"type": "Grid Failure", "impact": -0.15, "sentiment": -25, "conviction": -10, "desc": "Cascading power loss across metro regions"},
        {"type": "Civil Unrest", "impact": -0.07, "sentiment": -20, "conviction": -5, "desc": "Protests erupting in capital"},
        {"type": "Data Leak", "impact": -0.03, "sentiment": -12, "conviction": -8, "desc": "Exposing sensitive government communications"},
        {"type": "Supply Chain Collapse", "impact": -0.12, "sentiment": -18, "conviction": -5, "desc": "Essential goods stuck in transit"},
        {"type": "Algorithmic Panic", "impact": -0.08, "sentiment": -30, "conviction": -25, "desc": "HFT bots trigger flash crash"},

        # --- 3rd Order: System Shocks (Triggered by 2nd) ---
        {"type": "Regime Collapse", "impact": -0.4, "sentiment": -50, "conviction": -40, "desc": "Government dissolves under pressure"},
        {"type": "Martial Law", "impact": 0.1, "sentiment": -40, "conviction": 15, "desc": "Military assumes direct control"},
        {"type": "Technological Singularity", "impact": 0.2, "sentiment": 50, "conviction": 50, "desc": "AI assumes resource allocation"},
        {"type": "Total Isolation", "impact": -0.2, "sentiment": -30, "conviction": 30, "desc": "Borders sealed completely"},
        {"type": "Hyperinflation", "impact": -0.3, "sentiment": -45, "conviction": -30, "desc": "Currency becomes worthless"}
    ]

    # 3. Causal Graph (Event Chaining)
    # Event -> [ (NextEvent, Probability) ]
    causal_graph = {
        "Cyber Operation": [("Grid Failure", 0.4), ("Data Leak", 0.6), ("Algorithmic Panic", 0.3)],
        "Military Maneuver": [("Civil Unrest", 0.3), ("Total Isolation", 0.1)],
        "Trade Sanction": [("Supply Chain Collapse", 0.5), ("Currency Devaluation", 0.4)],
        "Grid Failure": [("Civil Unrest", 0.6), ("Martial Law", 0.2)],
        "Civil Unrest": [("Regime Collapse", 0.1), ("Martial Law", 0.4)],
        "Currency Devaluation": [("Hyperinflation", 0.2), ("Civil Unrest", 0.3)],
        "Liquidity Crisis": [("Algorithmic Panic", 0.7), ("Hyperinflation", 0.1)],
        "AI Hallucination": [("Algorithmic Panic", 0.5), ("Grid Failure", 0.1)],
        "Quantum Breakthrough": [("Technological Singularity", 0.05), ("Market Rally", 0.8)]
    }

    # 4. Probability Sets (Regimes)
    probability_sets = {
        "Standard": {"volatility": 0.05, "chain_multiplier": 1.0},
        "Crisis": {"volatility": 0.15, "chain_multiplier": 2.0}, # Higher chance of chains
        "Stagnation": {"volatility": 0.01, "chain_multiplier": 0.5},
        "Acceleration": {"volatility": 0.1, "chain_multiplier": 1.5}
    }

    # 5. Agent Profiles (Refined)
    agent_profiles = {
        "Conservative": {"bias": "Stability", "reaction": "Dampen"},
        "Aggressive": {"bias": "Expansion", "reaction": "Amplify"},
        "Chaotic": {"bias": "Entropy", "reaction": "Random"},
        "Analytic": {"bias": "Truth", "reaction": "Observe"}
    }

    simulation_config = {
        "default_iterations": 250, # Increased for better statistical significance
        "prediction_horizon_steps": 24, # 2 years effectively
        "stability_threshold": 0.25,
        "volatility_factor": 1.0
    }

    library = {
        "metadata": {
            "version": "2.0 (Enhanced)",
            "type": "Static Snapshot",
            "source": "Nexus Live Engine v2"
        },
        "sovereigns": sovereigns,
        "shadow_factions": shadow_factions,
        "event_templates": event_templates,
        "causal_graph": causal_graph,
        "probability_sets": probability_sets,
        "agent_profiles": agent_profiles,
        "simulation_config": simulation_config
    }

    try:
        with open(OUTPUT_PATH, 'w') as f:
            json.dump(library, f, indent=2)
        print(f"Successfully generated {OUTPUT_PATH}")
    except Exception as e:
        print(f"Error writing output file: {e}")

if __name__ == "__main__":
    main()
