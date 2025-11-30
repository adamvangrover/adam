# Setup Guide

This guide provides instructions for setting up your terminal and a "code wizard" to help you get started with the ADAM project.

## Terminal Setup

1.  **Install Python:** Make sure you have Python 3.8 or higher installed.
2.  **Clone the repository:** `git clone https://github.com/adamvangrover/adam.git`
3.  **Install dependencies:** `pip install -r requirements.txt`

## Code Wizard

The "code wizard" is a set of scripts that can help you to create new agents and other components of the ADAM system.

### Create a new agent

To create a new agent, run the following command:

```bash
python scripts/create_agent.py <agent_name>
```

This will create a new agent file in the `core/agents` directory with a basic template for a new agent.

### Create a new data source

To create a new data source, run the following command:

```bash
python scripts/create_data_source.py <data_source_name>
```

This will create a new data source file in the `core/data_sources` directory with a basic template for a new data source.
