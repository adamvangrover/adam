# Adam SLM: OAI-Aligned Training Pipeline Guide (v2)

**Protocol:** DeepMind Milestone Checkins
**Architecture:** Tinker-Based Neuro-Symbolic Optimization

## 1. Overview

This pipeline is designed to fine-tune the "Adam v23.5" Small Language Model (SLM) using "Artisanal" datasets. It emphasizes **Alignment** (OAI-style) and **Rigorous Project Management** (DeepMind-style).

## 2. Directory Structure

*   `tinker_lab/pipeline_v2/`
    *   `AGENTS.md`: The operational protocol and "Gates".
    *   `config.py`: Pydantic definitions for training jobs.
    *   `orchestrator.py`: The main runner script.
    *   `milestone_tracker.py`: System 2 logging utility.
    *   `logs/MILESTONES.md`: The immutable record of truth.

## 3. Setup

### Prerequisites
*   Python 3.10+
*   Dependencies: `pydantic` (and `tinker` for LIVE mode)

### Installation
```bash
# In the root repo
export PYTHONPATH=$PYTHONPATH:.
```

## 4. Usage

### Running a Mock Training Run
The system defaults to `MOCK` mode if no `TINKER_API_KEY` is present. This is perfect for verifying the "Gates" and data flow.

```bash
python3 tinker_lab/pipeline_v2/orchestrator.py
```

### Inspecting Milestones
After running, check the log:
```bash
cat tinker_lab/pipeline_v2/logs/MILESTONES.md
```

## 5. Alignment Philosophy

We follow a "Constitution" for Adam:
1.  **Truthfulness:** Grounding in "Artisanal" data (10-Ks, verified math).
2.  **Helpfulness:** Clear, structured output (JSON/Markdown).
3.  **Harmlessness:** Conservative risk assessment; do not invent data.

The `artisanal_training_data.json` contains weighted examples that reinforce these behaviors.
