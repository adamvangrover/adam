import { AdamNode, UUID } from "./core";

// ==============================================================================
// 16. Memory
// ==============================================================================

export interface Memory extends AdamNode {
  agentId: UUID;
  shortTermContextIds: UUID[];
  longTermKnowledgeIds: UUID[]; // References to Graph Nodes
}

export interface Reflection extends AdamNode {
  agentId: UUID;
  insight: string;
  sourceContextIds: UUID[];
}

export interface Goal extends AdamNode {
  agentId: UUID;
  description: string;
  isAchieved: boolean;
}
