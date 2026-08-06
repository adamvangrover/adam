import { AdamNode, UUID } from "./core";

// ==============================================================================
// 19. Scheduler
// ==============================================================================

export interface Schedule extends AdamNode {
  targetWorkflowId: UUID;
}

export interface Cron extends Schedule {
  expression: string;
}

export interface MarketOpen extends Schedule {
  exchangeId: UUID;
  offsetMinutes: number;
}

export interface MarketClose extends Schedule {
  exchangeId: UUID;
  offsetMinutes: number;
}

export interface EventDriven extends Schedule {
  topic: string;
  condition?: string;
}

export interface WebHook extends Schedule {
  endpoint: string;
}

export interface Manual extends Schedule {
  triggeredBy: UUID; // User ID
}
