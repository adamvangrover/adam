import { AdamNode, UUID } from "./core";

// ==============================================================================
// 10. Scenarios
// ==============================================================================

export interface Scenario extends AdamNode {
  name: string;
  assumptions: UUID[]; // References to Variables
  shockedVariables: Record<string, number>; // Use string key instead of UUID to satisfy Record signature constraint where key must be string/number/symbol. Values here represent shock magnitudes.
  probability: number;
}
