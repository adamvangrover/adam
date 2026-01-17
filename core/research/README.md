# Adam Research & Advanced Architectures

This directory contains experimental and advanced implementations of next-generation financial technology concepts, integrated into the Adam system.

## 1. Federated Learning (`core/research/federated_learning/`)

**Context:** Privacy-preserving distributed training for credit risk models across multiple institutions.

**Implementation:**
- **Coordinator:** `fl_coordinator.py` implements a standard `FedAvg` (Federated Averaging) algorithm.
- **Client:** `fl_client.py` simulates individual banks with private local data (synthetic credit profiles).
- **Model:** `model.py` defines a shared PyTorch Neural Network for credit scoring.

**Usage:**
```python
from core.research.federated_learning.fl_coordinator import FederatedCoordinator
coordinator = FederatedCoordinator(num_clients=5)
coordinator.run_round(1)
```

## 2. Graph Neural Networks (`core/research/gnn/`)

**Context:** Deep learning on the `UnifiedKnowledgeGraph` to detect systemic risks and hidden contagion paths.

**Implementation:**
- **Engine:** `engine.py` bridges the Knowledge Graph (NetworkX) with PyTorch, building adjacency matrices and node features.
- **Layer:** `layers.py` implements a custom Graph Convolutional Layer (GCN) supporting sparse matrix operations.
- **Model:** `model.py` defines a multi-layer GCN for node classification/risk scoring.

**Usage:**
```python
from core.research.gnn.engine import GraphRiskEngine
engine = GraphRiskEngine()
risk_scores = engine.predict_risk()
```

## 3. One-Shot World Models (`core/research/oswm/`)

**Context:** Model-Based Reinforcement Learning using Transformers to predict market dynamics and generate counterfactual scenarios via "One-Shot" in-context learning.

**Implementation:**
- **Inference:** `inference.py` manages the "Pre-training" on synthetic physics priors and "In-Context" generation on real market data.
- **Transformer:** `transformer.py` implements a causal Transformer architecture for sequence modeling.

**Usage:**
```python
from core.research.oswm.inference import OSWMInference
oswm = OSWMInference()
oswm.pretrain_on_synthetic_prior()
prediction = oswm.generate_scenario(market_context)
```

## Integration

Run the demo script to see all modules in action:
```bash
python scripts/run_research_demo.py
```
Output is saved to `showcase/data/research_output.json`.
