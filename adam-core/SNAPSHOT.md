# SNAPSHOT: adam-core (retained in place)

**Purpose:** Rust deterministic kernel — fixed-point `BasisPoints` arithmetic, SHA-256 hashing, and the memory subsystem (`src/memory/`: admission, canonicalization, provenance, retrieval).

**Status:** Not archived. It is the only Rust memory implementation in the repo, so it stays at `adam-core/` as an optional runtime slice (see `REPO_MAP.md` §6).

**Build / test:** `cargo test --manifest-path adam-core/Cargo.toml` (no Python bindings; standalone crate).
