import { UUID, Timestamp } from '../types';

export interface User {
    id: UUID;
    username: string;
    roles: string[];
}

export interface Session {
    id: UUID;
    userId: UUID;
    startedAt: Timestamp;
    expiresAt?: Timestamp;
}

export interface Permissions {
    roles: string[];
    scopes: string[];
}

export interface Environment {
    env: 'development' | 'staging' | 'production' | 'test';
    variables: Record<string, string>;
}

export interface RuntimeContext {
    user: User;
    session: Session;
    permissions: Permissions;
    environment: Environment;
    graph: any; // Placeholder for GraphStore
    cache: any; // Placeholder for Cache interface
    clock: any; // Placeholder for Clock
    logger: any; // Placeholder for Logger
    trace: any; // Placeholder for Tracing
    messageBus: any; // Placeholder for Event/Message Bus
    scheduler: any; // Placeholder for Scheduler
    plugins: any; // Placeholder for PluginHost
    featureStore: any; // Placeholder for FeatureStore
}
