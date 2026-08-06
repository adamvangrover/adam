import { UUID, Timestamp, ImmutableNode } from '../src/types';
import { Entity, Variable, Observation, Relationship, Edge, ModelOutput, Decision } from '../src/graph';
import { Dataset, Feature, FeatureState } from '../src/datasets';
import { Environment, RuntimeContext, Session, User, Permissions } from '../src/environment';

describe('Core Types Verification', () => {
    it('should define basic types correctly', () => {
        const id: UUID = '1234-5678';
        expect(id).toBe('1234-5678');

        const now: Timestamp = Date.now();
        expect(typeof now).toBe('number');

        const node: ImmutableNode = {
            parentVersion: 'prev-123',
            branch: 'main',
            commit: 'abc',
            author: 'jules',
            reason: 'initial'
        };
        expect(node.parentVersion).toBe('prev-123');
    });

    it('should create an Entity correctly', () => {
        const entity: Entity = {
            id: 'e1',
            type: 'Entity',
            name: 'CompanyA',
            properties: { sector: 'Tech' },
            createdAt: Date.now(),
            updatedAt: Date.now(),
            parentVersion: 'v0'
        };
        expect(entity.type).toBe('Entity');
        expect(entity.parentVersion).toBe('v0');
    });

    it('should create a Feature correctly', () => {
        const state: FeatureState = 'Cleaned';
        const feature: Feature = {
            id: 'f1',
            datasetId: 'd1',
            name: 'Revenue',
            state: state,
            dataType: 'number'
        };
        expect(feature.state).toBe('Cleaned');
    });

    it('should type check RuntimeContext correctly', () => {
        const user: User = { id: 'u1', username: 'admin', roles: ['admin'] };
        const session: Session = { id: 's1', userId: 'u1', startedAt: Date.now() };
        const perms: Permissions = { roles: ['admin'], scopes: ['*'] };
        const env: Environment = { env: 'development', variables: {} };

        const ctx: RuntimeContext = {
            user,
            session,
            permissions: perms,
            environment: env,
            graph: {}, cache: {}, clock: {}, logger: {}, trace: {}, messageBus: {}, scheduler: {}, plugins: {}, featureStore: {}
        };

        expect(ctx.user.username).toBe('admin');
        expect(ctx.environment.env).toBe('development');
    });
});
