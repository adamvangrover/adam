import { AdamNode, UUID, ISO8601Date } from "./core";

// ==============================================================================
// 6. Observations
// ==============================================================================

export interface Observation extends AdamNode {
  variableId: UUID;
  timestamp: ISO8601Date;
  sourceId: UUID; // Originator (Model or Dataset ID)
  confidence: number;
}

export interface NumericObservation extends Observation {
  value: number;
}

export interface CategoricalObservation extends Observation {
  value: string;
}
