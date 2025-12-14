# Copilot Quest: Gamified Training Blueprint

## 1. Overview
"Copilot Quest" is an interactive game designed for corporate training workshops. It gamifies the process of learning how to work with the Adam AI system, while simultaneously capturing high-value human-in-the-loop data for model fine-tuning.

## 2. Objectives
*   **User Training:** Teach analysts how to formulate effective prompts and interpret AI outputs.
*   **Data Capture:** Collect "Gold Standard" corrections and reasoning traces from human experts.
*   **Stress Testing:** Identify edge cases where the AI fails by incentivizing users to "break" the system.

## 3. Game Mechanics
### Phase 1: The Detective (Information Retrieval)
*   **Goal:** Find specific, obscure facts within a massive dataset (e.g., "Find the exact date of the CEO's stock option grant in 2021").
*   **Scoring:** Speed + Accuracy.
*   **Data Captured:** Search queries, relevance feedback.

### Phase 2: The Auditor (Fact Checking)
*   **Goal:** The AI generates a "hallucinated" credit memo. The user must identify and correct the errors.
*   **Scoring:** Number of errors found + Quality of correction.
*   **Data Captured:** DPO (Direct Preference Optimization) pairs: {Bad Output, Good Output}.

### Phase 3: The Red Team (Adversarial)
*   **Goal:** Trick the AI into approving a bad loan or violating a compliance rule.
*   **Scoring:** Successfully bypassing a guardrail earns maximum points.
*   **Data Captured:** Vulnerabilities for the "Bear" agent to learn from.

## 4. Technical Architecture
*   **Frontend:** React-based UI with a "Quest Log" and real-time leaderboards.
*   **Backend:** FastAPI service recording all interactions to the `dpo_dataset` table.
*   **Analytics:** Dashboard showing common user errors and AI failure modes.

## 5. Integration with RLHF
All data collected during "Copilot Quest" sessions is automatically tagged and pipelined into the `train_dpo.py` workflow for the next model training cycle.
