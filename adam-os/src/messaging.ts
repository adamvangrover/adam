import { AdamNode, JSONValue } from "./core";

// ==============================================================================
// 22. Messaging & Event Bus
// ==============================================================================

export interface EventMessage extends AdamNode {
  topic: string;
  payload: JSONValue;
}

export interface EventBus extends AdamNode {
  name: string;
  topics: string[];
}
