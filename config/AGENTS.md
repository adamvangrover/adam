# Configuration Files

This directory contains the configuration files for the ADAM system. Each file controls a specific aspect of the system's behavior.

## File Overview

*   **`api.yaml`:** Configuration for external APIs.
*   **`config.yaml`:** General configuration for the ADAM system.
*   **`knowledge_graph.yaml`:** Configuration for the knowledge graph.
*   **`logging.yaml`:** Configuration for the logging system.
*   **`reporting.yaml`:** Configuration for the reporting system.
*   **`settings.yaml`:** General settings for the ADAM system.

## Detailed Configuration Options

### `api.yaml`

This file contains the API keys and other credentials for accessing external APIs.

**Example:**

```yaml
news_api:
  api_key: "YOUR_API_KEY"
  url: "https://api.example.com/news"
```

### `config.yaml`

This file contains the general configuration for the ADAM system, such as the list of active agents and the default settings for the system.

**Example:**

```yaml
active_agents:
  - "market_sentiment_agent"
  - "fundamental_analyst_agent"

default_settings:
  log_level: "INFO"
  max_threads: 10
```

### `knowledge_graph.yaml`

This file contains the configuration for the knowledge graph, including the connection details for the graph database and the schema for the graph.

**Example:**

```yaml
connection:
  host: "localhost"
  port: 7687
  username: "neo4j"
  password: "YOUR_PASSWORD"

schema:
  nodes:
    - label: "Company"
      properties:
        - name: "name"
          type: "string"
        - name: "ticker"
          type: "string"
  relationships:
    - type: "HAS_CEO"
      start_node: "Company"
      end_node: "Person"
```

### `logging.yaml`

This file contains the configuration for the logging system, including the log level, the log format, and the log output.

**Example:**

```yaml
version: 1
formatters:
  brief:
    format: "%(asctime)s - %(levelname)s - %(message)s"
handlers:
  console:
    class: logging.StreamHandler
    formatter: brief
    level: INFO
root:
  handlers: [console]
  level: INFO
```

### `reporting.yaml`

This file contains the configuration for the reporting system, including the report templates and the output formats.

**Example:**

```yaml
templates:
  daily_briefing: "templates/daily_briefing.md"
  weekly_summary: "templates/weekly_summary.md"

outputs:
  - type: "pdf"
    path: "reports/daily_briefing.pdf"
  - type: "html"
    path: "reports/weekly_summary.html"
```

### `settings.yaml`

This file contains general settings for the ADAM system, such as the paths to the data and log directories.

**Example:**

```yaml
data_dir: "data/"
log_dir: "logs/"
```

## Modifying Configuration Files

When modifying configuration files, please ensure that you understand the impact of your changes. Incorrectly configured files can cause the system to behave unexpectedly.

### Best Practices

*   **Backup:** Before making any changes, create a backup of the original file.
*   **Documentation:** Refer to the documentation for each configuration file to understand the available options and their valid values.
*   **Validation:** After making changes, validate the configuration to ensure that it is syntactically correct and that the values are within the expected ranges.

## Adding New Configuration Files

When adding a new configuration file, please follow these steps:

1.  **Create the file:** Create a new YAML file in this directory.
2.  **Define the schema:** Define the schema for the configuration file, including the available options and their valid values.
3.  **Update the documentation:** Update the documentation to include the new configuration file and its options.
4.  **Implement the loading logic:** Implement the logic to load the configuration file and make it available to the system.

By following these guidelines, you can help to ensure that the ADAM system remains stable and easy to configure.
