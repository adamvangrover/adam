import { AdamNode, UUID, JSONValue } from "./core";

// ==============================================================================
// 17. Tools
// ==============================================================================

export interface ToolDefinition extends AdamNode {
  name: string;
  description: string;
  expectedInputs: string[];
  expectedOutputs: string[];
}

export interface ToolVersion extends AdamNode {
  toolId: UUID;
  tag: string;
  entryPoint: string;
}

export interface ToolAccess extends AdamNode {
  agentId: UUID;
  toolId: UUID;
}

export interface ToolInvocation extends AdamNode {
  toolVersionId: UUID;
  invokedBy: UUID; // Agent or User ID
  parameters: Record<string, JSONValue>;
}

export interface ToolResult extends AdamNode {
  invocationId: UUID;
  status: "Success" | "Failure";
  output: JSONValue;
}
