# SNAPSHOT: tinker_lab

**Purpose:** R&D lab for data generation / model training with the `tinker-cookbook` submodule, plus one-off HTML/JS/lint patch scripts and their logs.

**Notable implementations**
- `01_Data_Generation.ipynb`, `02_Model_Training.ipynb` — fine-tuning workflow.
- `pipeline_v2/` — milestone-tracked training orchestrator.
- `tinker-cookbook/` — git submodule (path updated in `.gitmodules`); `scripts/extract_extended_data.py` reads its `example-data/`.
- `fix_*.py`, `patch_*.py`, `update_script_*.py` — one-off repo-maintenance scripts (copies also in `archive/`).

**Pulled:** nothing (no runtime code). **Not imported** by live code.
