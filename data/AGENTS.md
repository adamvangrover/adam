# Data Files

This directory contains the data files used by the ADAM system. These files include datasets for training and testing, as well as knowledge bases and other resources.

## Data Schemas

Here are the schemas for some of the most important data files in this directory:

### `company_data.json`

This file contains fundamental data for a list of companies. The file is a JSON array, where each object represents a company and has the following schema:

```json
{
  "name": "string",
  "ticker": "string",
  "sector": "string",
  "market_cap": "number",
  "revenue": "number",
  "net_income": "number"
}
```

### `knowledge_graph.json`

This file contains the knowledge graph for the ADAM system. The file is a JSON object that represents the graph in a node-link format.

**Nodes:**

*   **`id`:** A unique identifier for the node.
*   **`label`:** The label of the node (e.g., "Company", "Person").
*   **`properties`:** A JSON object containing the properties of the node.

**Links:**

*   **`source`:** The ID of the source node.
*   **`target`:** The ID of the target node.
*   **`type`:** The type of the relationship between the two nodes (e.g., "HAS_CEO", "WORKS_AT").

### `market_data.csv`

This file contains historical market data for a list of stocks. The file is a CSV file with the following columns:

*   **`date`:** The date of the market data.
*   **`ticker`:** The ticker symbol of the stock.
*   **`open`:** The opening price of the stock.
*   **`high`:** The highest price of the stock during the day.
*   **`low`:** The lowest price of the stock during the day.
*   **`close`:** The closing price of the stock.
*   **`volume`:** The trading volume of the stock.

## File Formats

The data files are stored in a variety of formats, including:

*   **`json`:** JavaScript Object Notation. A lightweight data-interchange format that is easy for humans to read and write and easy for machines to parse and generate.
*   **`jsonld`:** JSON for Linking Data. An extension of JSON that provides a way to create machine-readable data on the web.
*   **`csv`:** Comma-Separated Values. A text file in which values are separated by commas.
*   **`ttl`:** Terse RDF Triple Language. A format for expressing RDF data in a compact and human-readable way.
*   **`jsonl`:** JSON Lines. A format for storing structured data that may be processed one record at a time.

## Adding New Data Files

When adding new data files, please follow these steps:

1.  **Choose the appropriate file format.** The file format should be chosen based on the type of data and how it will be used.
2.  **Add the file to this directory.**
3.  **Update the documentation.** If the new data file is used in a specific part of the system, please update the relevant documentation to reflect this.

## Data Integrity

It is important to maintain the integrity of the- data files in this directory. Before making any changes to a data file, please ensure that you understand the impact of your changes.

By following these guidelines, you can help to ensure that the data used by the ADAM system is accurate, up-to-date, and well-maintained.

## Data Usage

This section provides examples of how to load and use the data files in Python.

### Loading JSON Files

To load a JSON file, you can use the `json` module in the Python standard library.

```python
import json

with open("data/company_data.json", "r") as f:
    company_data = json.load(f)

for company in company_data:
    print(company["name"])
```

### Loading CSV Files

To load a CSV file, you can use the `csv` module in the Python standard library.

```python
import csv

with open("data/market_data.csv", "r") as f:
    reader = csv.reader(f)
    for row in reader:
        print(row)
```

### Loading JSONL Files

To load a JSONL file, you need to read the file line by line and parse each line as a JSON object.

```python
import json

with open("data/simulated_JSONL_output.jsonl", "r") as f:
    for line in f:
        data = json.loads(line)
        print(data)
```
=======
# Data Files

This directory contains the data files used by the ADAM system. These files include datasets for training and testing, as well as knowledge bases and other resources.

## Data Schemas

Here are the schemas for some of the most important data files in this directory:

### `company_data.json`

This file contains fundamental data for a list of companies. The file is a JSON array, where each object represents a company and has the following schema:

```json
{
  "name": "string",
  "ticker": "string",
  "sector": "string",
  "market_cap": "number",
  "revenue": "number",
  "net_income": "number"
}
```

### `knowledge_graph.json`

This file contains the knowledge graph for the ADAM system. The file is a JSON object that represents the graph in a node-link format.

**Nodes:**

*   **`id`:** A unique identifier for the node.
*   **`label`:** The label of the node (e.g., "Company", "Person").
*   **`properties`:** A JSON object containing the properties of the node.

**Links:**

*   **`source`:** The ID of the source node.
*   **`target`:** The ID of the target node.
*   **`type`:** The type of the relationship between the two nodes (e.g., "HAS_CEO", "WORKS_AT").

### `market_data.csv`

This file contains historical market data for a list of stocks. The file is a CSV file with the following columns:

*   **`date`:** The date of the market data.
*   **`ticker`:** The ticker symbol of the stock.
*   **`open`:** The opening price of the stock.
*   **`high`:** The highest price of the stock during the day.
*   **`low`:** The lowest price of the stock during the day.
*   **`close`:** The closing price of the stock.
*   **`volume`:** The trading volume of the stock.

## File Formats

The data files are stored in a variety of formats, including:

*   **`json`:** JavaScript Object Notation. A lightweight data-interchange format that is easy for humans to read and write and easy for machines to parse and generate.
*   **`jsonld`:** JSON for Linking Data. An extension of JSON that provides a way to create machine-readable data on the web.
*   **`csv`:** Comma-Separated Values. A text file in which values are separated by commas.
*   **`ttl`:** Terse RDF Triple Language. A format for expressing RDF data in a compact and human-readable way.
*   **`jsonl`:** JSON Lines. A format for storing structured data that may be processed one record at a time.

## Adding New Data Files

When adding new data files, please follow these steps:

1.  **Choose the appropriate file format.** The file format should be chosen based on the type of data and how it will be used.
2.  **Add the file to this directory.**
3.  **Update the documentation.** If the new data file is used in a specific part of the system, please update the relevant documentation to reflect this.

## Data Integrity

It is important to maintain the integrity of the- data files in this directory. Before making any changes to a data file, please ensure that you understand the impact of your changes.

By following these guidelines, you can help to ensure that the data used by the ADAM system is accurate, up-to-date, and well-maintained.
