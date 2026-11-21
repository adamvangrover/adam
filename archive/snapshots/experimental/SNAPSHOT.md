# SNAPSHOT: experimental

**Purpose:** Path B lab code: inference experiments, QMC simulation, v23 microservice scaffolding, NEXUS-AURORA concept, SLM distillation.

**Notable implementations**
- `inference_lab/models/kv_cache.py`, `inference_lab/reasoning/tree_of_thoughts.py` — inference primitives.
- `qmc/qms.py` — quasi-Monte-Carlo simulation sketch.
- `v23_scaffolding/` — non-functional microservice scaffolding (k8s, GNN, DSPy, Go consumer).
- `adamos_kernel/` — older copy; the CI-tested kernel lives at `core/experimental/adamos_kernel/`.
- `ml_forward_outlook.py`, `slm_distillation/`, `nexus_aurora/`.

**Pulled:** nothing. `core/experimental/` is a separate live package and is not part of this snapshot.
**Not imported** by live code.
