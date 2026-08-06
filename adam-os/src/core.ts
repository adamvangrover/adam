// ==============================================================================
// 1. Core Primitives & Base Types
// ==============================================================================

export type JSONValue =
  | string
  | number
  | boolean
  | null
  | JSONValue[]
  | { [key: string]: JSONValue };

export type UUID = string;
export type ISO8601Date = string;

export interface AdamNode {
  id: UUID;
  createdAt: ISO8601Date;
  updatedAt: ISO8601Date;
  version: string;
  metadata?: Record<string, JSONValue>;
}
