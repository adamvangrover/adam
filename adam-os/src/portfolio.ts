import { AdamNode, UUID } from "./core";

// ==============================================================================
// 12. Portfolio
// ==============================================================================

export interface Security extends AdamNode {
  ticker: string;
  assetClass: string;
}

export interface Position extends AdamNode {
  securityId: UUID;
  quantity: number;
  marketValue: number;
  weight: number;
}

export interface Portfolio extends AdamNode {
  name: string;
  positions: Position[];
  benchmarkId?: UUID;
  riskMetricsId?: UUID; // Reference to structured risk node
}

export interface Benchmark extends AdamNode {
  name: string;
  constituents: Position[];
}
