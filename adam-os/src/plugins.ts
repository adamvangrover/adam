import { AdamNode } from "./core";

// ==============================================================================
// 23. Plugins
// ==============================================================================

export interface Plugin extends AdamNode {
  name: string;
  pluginVersion: string;
  entryPoint: string; // Path to logic
}
