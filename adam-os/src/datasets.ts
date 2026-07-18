import { AdamNode, UUID } from "./core";

// ==============================================================================
// 4. Dataset Layer
// ==============================================================================

export interface DataSourceType extends AdamNode {
  protocol: string; // e.g., "REST", "GraphQL", "S3", "SQL"
  format: string; // e.g., "JSON", "CSV", "Parquet"
}

export interface IngestionVector extends AdamNode {
  sourceTypeId: UUID;
  frequency: string; // e.g., "Daily", "Intraday", "Streaming"
  endpointUrl?: string;
}

export interface Dataset extends AdamNode {
  name: string;
  description: string;
  ingestionVectorId: UUID;
  schemaId?: UUID;
}
