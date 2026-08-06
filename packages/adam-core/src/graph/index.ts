import { UUID, Timestamp, ImmutableNode } from '../types';

export interface BaseNode extends ImmutableNode {
    id: UUID;
    type: string;
    createdAt: Timestamp;
    updatedAt: Timestamp;
}

export interface Entity extends BaseNode {
    type: 'Entity';
    name: string;
    properties: Record<string, any>;
}

export interface Variable extends BaseNode {
    type: 'Variable';
    entityId: UUID;
    name: string;
    dataType: string;
}

export interface Observation extends BaseNode {
    type: 'Observation';
    variableId: UUID;
    timestamp: Timestamp;
    value: any;
}

export interface Edge extends BaseNode {
    type: 'Edge' | 'Relationship';
    sourceId: UUID;
    targetId: UUID;
    relationshipType: string;
    weight?: number;
    properties?: Record<string, any>;
}

export interface Relationship extends Edge {
    type: 'Relationship';
}

export interface ModelOutput extends BaseNode {
    type: 'ModelOutput';
    modelId: UUID;
    runId: UUID;
    outputs: Record<string, any>;
}

export interface Decision extends BaseNode {
    type: 'Decision';
    contextId: UUID;
    decisionData: Record<string, any>;
    confidence?: number;
}
