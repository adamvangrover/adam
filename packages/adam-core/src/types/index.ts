export type UUID = string;
export type Timestamp = number | string | Date;

export interface VersionedNode {
    parentVersion?: UUID;
    branch?: string;
    commit?: string;
    author?: string;
    reason?: string;
}

export type Immutable<T> = {
    readonly [P in keyof T]: T[P];
};

export type ImmutableNode = Immutable<VersionedNode>;
