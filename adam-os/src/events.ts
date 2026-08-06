import { AdamNode, UUID, JSONValue } from "./core";

// ==============================================================================
// 26. Event Sourcing
// ==============================================================================

export interface NodeEvent extends AdamNode {
  targetNodeId: UUID;
  eventType: "Created" | "Updated" | "Deleted" | "Published" | "Approved" | "Rejected";
  payload?: JSONValue;
}
