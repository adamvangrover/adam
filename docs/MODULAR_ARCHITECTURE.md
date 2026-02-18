# ADAM v24.0 Modular Architecture Guide

## Overview
The **Modular Data Loading Architecture** represents a strategic shift from monolithic data bundles to an on-demand, asynchronous loading model. This approach enables "Lite" deployments, faster initial load times, and better resource management for the ADAM platform.

## Key Components

### 1. Data Extraction (`scripts/extract_seed_data.py`)
This utility script is responsible for decoupling large datasets from the main application bundle.
- **Input:** `showcase/js/mock_data.js` (The monolithic data source)
- **Output:**
    - `showcase/data/seed_reports.json`: Comprehensive list of intelligence reports.
    - `showcase/data/seed_credit_memos.json`: Deep simulated credit risk artifacts.
    - `showcase/data/seed_file_index.json`: Full repository file tree for navigation.

### 2. Modular Data Manager (`showcase/js/data_manager_modular.js`)
A centralized JavaScript class `ModularDataManager` that handles fetching and caching.
- **Usage:**
  ```javascript
  // Load reports
  window.modularDataManager.loadReports().then(data => { ... });

  // Clear cache
  window.modularDataManager.clearCache();
  ```
- **Features:**
    - **Caching:** Prevents redundant network requests.
    - **Error Handling:** Gracefully manages fetch failures.
    - **Performance:** Asynchronous execution prevents UI blocking.

### 3. Modular Dashboard (`showcase/modular_dashboard.html`)
A Proof-of-Concept (POC) interface demonstrating the architecture.
- **Visualizer:** Dynamically renders loaded data into Cards or Lists.
- **Filtering:** Client-side search for instant data retrieval.
- **Security:** Uses `textContent` to prevent XSS vulnerabilities when rendering user data.

## Deployment & Portability
This architecture is fully integrated with the **ADAM Module Exporter** (`scripts/export_module.py`).
- **Command:** `python3 scripts/export_module.py modular_dashboard`
- **Result:** Generates a standalone, portable version of the dashboard in `exports/modular_dashboard/`, including all necessary JSON data and scripts.

## Future Evolution
- **Lazy Loading:** Implement intersection observers to load data chunks only when scrolled into view.
- **Versioning:** Add version hashes to JSON files to ensure cache consistency during updates.
- **Compression:** Enable GZIP/Brotli support for the JSON artifacts to further reduce bandwidth.
