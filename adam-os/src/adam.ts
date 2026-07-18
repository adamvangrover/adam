import { AdamNode, UUID } from "./core";
import { Variable, GraphEdge } from "./graph";
import { Dataset } from "./datasets";
import { Model } from "./models";
import { Environment } from "./environment";
import { Scenario } from "./scenarios";
import { Portfolio } from "./portfolio";
import { Agent } from "./agents";
import { Workflow } from "./execution";
import { Plugin } from "./plugins";

// ==============================================================================
// 27. Root Composition
// ==============================================================================

export interface GraphStore {
  variables: Variable[];
  edges: GraphEdge[];
}

export interface AdamOS extends AdamNode {
  graph: GraphStore;
  datasets: Dataset[];
  models: Model[];
  environments: Environment[];
  scenarios: Scenario[];
  portfolios: Portfolio[];
  agents: Agent[];
  workflows: Workflow[];
  plugins: Plugin[];
}
