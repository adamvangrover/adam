# Versioning and Migration Guide

This document outlines the versioning scheme for data files in the ADAM system and provides instructions for migrating data between versions. For a high-level overview of the data, see the [Data Navigation Guide](data/DATA_NAVIGATION.md).

## 1. Versioning Scheme

Data files are versioned using a semantic versioning scheme: `MAJOR.MINOR.PATCH`.

*   **MAJOR:** Incremented for incompatible API changes.
*   **MINOR:** Incremented for adding functionality in a backwards-compatible manner.
*   **PATCH:** Incremented for backwards-compatible bug fixes.

The version number for each data file is stored in the `version_control.json` file in the root directory.

## 2. Change Log

### 2.1. `knowledge_base.json`

*   **2.0.0 (2024-07-30):**
    *   Added a new `Industry` section.
    *   Renamed `Valuation` to `ValuationMethods`.
*   **1.1.0 (2024-07-29):**
    *   Added a new `ESG` subsection to the `RiskManagement` section.
*   **1.0.0 (2024-07-28):**
    *   Initial version.

## 3. Automated Versioning

The `scripts/version_data.py` script can be used to automatically increment the version number of a data file and to add an entry to the change log in this file.

### 3.1. Usage

```bash
python scripts/version_data.py <file_path> <version_type> <change_description>
```

*   `<file_path>`: The path to the data file.
*   `<version_type>`: One of `major`, `minor`, or `patch`.
*   `<change_description>`: A description of the change.

## 4. Data Migration

The `scripts/migration` directory contains scripts for migrating data from one version to another.

### 4.1. `knowledge_base.json`: 1.1.0 to 2.0.0

The `scripts/migration/migrate_knowledge_base_1.1.0_to_2.0.0.py` script migrates the `knowledge_base.json` file from version 1.1.0 to 2.0.0.

### 4.2. Usage

```bash
python scripts/migration/migrate_knowledge_base_1.1.0_to_2.0.0.py
```

## 5. Developer Notes

*   When making changes to a data file, be sure to update the version number and to add an entry to the change log.
*   If the change is not backwards-compatible, you will also need to create a migration script.

## 6. Future Development

*   **Automated Migration:** We are exploring ways to automate the data migration process.
*   **Data Rollbacks:** We also plan to implement a data rollback feature that will allow us to revert to a previous version of a data file in the event of a problem.
