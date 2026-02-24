# Versioning Strategy & Release Cycle

## Overview
Adam follows [Semantic Versioning 2.0.0](https://semver.org/). Given the mission-critical nature of financial systems, strict adherence to versioning contracts is enforced to prevent regressions in live trading or risk environments.

## Version Format
`MAJOR.MINOR.PATCH`

- **MAJOR**: Incompatible API changes or fundamental architectural shifts (e.g., v24.0 -> v26.0).
- **MINOR**: Additive functionality in a backward-compatible manner (e.g., adding a new agent or data source).
- **PATCH**: Backward-compatible bug fixes (e.g., fixing a prompt hallucination or a retry logic error).

## Branching Strategy

We utilize a modified **GitFlow** workflow:

- **`main`**: The stable, production-ready branch. All commits here are tagged releases.
- **`develop`**: The integration branch for features. Automated tests run on every push.
- **`feature/*`**: Individual feature branches (e.g., `feature/new-risk-agent`).
- **`hotfix/*`**: Urgent fixes for production (merged to `main` and `develop`).
- **`release/*`**: Preparation for a new production release (e.g., `release/v26.1.0`).

## Release Process

1.  **Freeze**: Code freeze on `develop`.
2.  **Test**: Run full integration suite (`pytest tests/integration`).
3.  **Bump**: Update `pyproject.toml` and `VERSION` file.
4.  **Tag**: Create a git tag (e.g., `v26.1.0`).
5.  **Deploy**: CI/CD pipeline deploys to staging, then production.

## Deprecation Policy

- **Deprecation Warning**: Features to be removed in the next Major version will emit a `DeprecationWarning`.
- **Grace Period**: At least one Minor version cycle is provided before removal.
