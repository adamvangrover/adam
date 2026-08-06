import { UUID, Timestamp, ImmutableNode } from '../types';

export interface Dataset extends ImmutableNode {
    id: UUID;
    name: string;
    description?: string;
    sourceUri?: string;
    createdAt: Timestamp;
    updatedAt: Timestamp;
}

export type FeatureState = 'Raw' | 'Cleaned' | 'Derived' | 'Normalized' | 'Embedded';

export interface Feature extends ImmutableNode {
    id: UUID;
    datasetId: UUID;
    name: string;
    state: FeatureState;
    dataType: string;
    description?: string;
}
