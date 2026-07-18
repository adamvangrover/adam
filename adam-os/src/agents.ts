import { AdamNode, UUID } from "./core";

// ==============================================================================
// 15. Agents
// ==============================================================================

export interface Capability extends AdamNode {
  name: string;
  description: string;
}

export interface Role extends AdamNode {
  name: string;
  capabilities: UUID[];
}

export interface Agent extends AdamNode {
  name: string;
  roleId: UUID;
  memoryId?: UUID;
}

export interface Task extends AdamNode {
  agentId: UUID;
  description: string;
  status: "Pending" | "InProgress" | "Completed" | "Failed";
}

export interface Conversation extends AdamNode {
  participants: UUID[]; // Agent or User IDs
  messages: UUID[]; // References to Message nodes
}
