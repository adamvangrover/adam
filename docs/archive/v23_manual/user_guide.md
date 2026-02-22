# Adam v23.0 User Guide

## Overview
Adam v23.0 introduces the **Adaptive System** architecture, moving beyond simple task execution to complex, cyclical reasoning. This guide explains how to interact with the new capabilities.

## Interaction Modes

The system automatically routes your query based on its complexity and intent. You do not need to specify a mode; simply ask your question naturally.

### 1. Neuro-Symbolic Planning (General Analysis)
**Intent:** "Analyze", "Plan", "Risk Assessment"
**Description:** The system dynamically constructs a workflow (graph) to answer open-ended questions.
**Examples:**
*   "Analyze the credit risk of Apple Inc. considering recent iPhone sales."
*   "Draft a strategy for entering the Asian market."

### 2. Red Team Simulation (Adversarial Testing)
**Intent:** "Attack", "Simulate Scenario", "Stress Test"
**Description:** The system adopts an adversarial persona to find weaknesses in a target entity or strategy.
**Examples:**
*   "Simulate a cyber attack on our payment gateway."
*   "Stress test the portfolio against a sudden interest rate hike."

### 3. ESG Analysis (Sustainability)
**Intent:** "ESG", "Green", "Sustainability"
**Description:** The system evaluates Environmental, Social, and Governance factors, checking for greenwashing and controversies.
**Examples:**
*   "What is the ESG score for Exxon Mobil?"
*   "Analyze the sustainability report of Tesla."

### 4. Regulatory Compliance (RegTech)
**Intent:** "Compliance", "KYC", "AML", "Regulation"
**Description:** The system checks an entity against jurisdictional regulations (Basel III, GDPR, etc.).
**Examples:**
*   "Check Generic Bank for Basel III compliance."
*   "Is this crypto transaction a potential AML violation?"

## Interpreting Results

The v23 system provides transparent reasoning steps ("Explainable AI").

*   **Human Readable Status:** Updates like "I am currently verifying the debt ratio..." appear in real-time.
*   **Critique Loop:** You may see the system "critique" its own work and "revise" it. This is normal and indicates the system is double-checking its facts.
*   **Final Report:** The output is usually a structured report with a clear conclusion or risk rating.

## Advanced Usage: The Dashboard

Visit the **Mission Control Dashboard** (`index.html`) to visualize the active graphs.
*   **Graph View:** See the nodes (circles) and edges (lines) light up as the system executes.
*   **State Inspection:** Click on a node to see the data (variables) at that point in time.

## Troubleshooting

*   **"Failed to generate plan":** The Planner could not map your query to known tools. Try rephrasing with clearer financial terms.
*   **"Dependency Error":** Ensure you have the `v23` dependencies installed (`pip install -r requirements.txt`).
