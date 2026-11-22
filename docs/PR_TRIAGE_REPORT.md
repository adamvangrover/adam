# Open PR Triage Report

Snapshot of open pull requests in `adamvangrover/adam` against `main` @ `58d88333`, taken 2026-11-22.
Read-only: nothing has been closed, merged or rebased. Every action below needs human sign-off.

## Method

- **Mergeability:** `git merge-tree --write-tree origin/main <head>` for every head branch (GitHub's own `mergeable` field was `UNKNOWN` for most PRs).
- **Deleted paths:** a file counts as deleted when neither it nor its parent directory exists on `main` (the #1436/#1437 reorganisation moved many trees).
- **Duplicates:** titles normalised (version numbers, dates and punctuation stripped); the newest PR with a given title is kept.
- **Categories:** P0 — rebase & merge (clean merge, ≤ 600 changed lines). P1 — salvage / split (valuable but conflicting or oversized). P2 — superseded / stale (close).

## Summary

| Category | Count |
|---|---|
| P0 | 8 |
| P1 | 85 |
| P2 | 118 |
| **Total** | 211 |

## Top 10 P0 PRs in merge order

Order: security first, then fixes, then dependency bumps, then features; oldest first within each group. Re-run CI on each after the previous one merges.

| # | PR | Title | Author | Changed lines |
|---|---|---|---|---|
| 1 | #1052 | Fix syntax error in swarm_runner.py causing CI failure | adamvangrover | 348 |
| 2 | #1417 | fix(ci): resolve Bandit B604 false positive and configure Ruff to extend pyproject.toml | adamvangrover | 4 |
| 3 | #728 | Bump actions/configure-pages from 3 to 5 | dependabot | 2 |
| 4 | #729 | Update rand requirement from 0.8 to 0.10 in /core/experimental/adamos_kernel | dependabot | 2 |
| 5 | #1433 | Bump react-dom and @types/react-dom in /services/webapp/client | dependabot | 127 |
| 6 | #621 | Expand Governance Layer: Tests, Middleware, and Documentation | adamvangrover | 206 |
| 7 | #1196 | Add Market Mayhem Vol 5 HTML Report | adamvangrover | 559 |
| 8 | #1199 | feat(anomaly): implement LOF, One-Class SVM and ratio-based anomaly detection | adamvangrover | 66 |

## P0 — Rebase & merge

| PR | Title | Author | Created | Merge vs main | Lines | Files | Reason |
|---|---|---|---|---|---|---|---|
| #621 | Expand Governance Layer: Tests, Middleware, and Documentation | adamvangrover | 2026-02-15 | clean | 206 | 3 | merges cleanly, small diff |
| #728 | Bump actions/configure-pages from 3 to 5 | dependabot | 2026-02-25 | clean | 2 | 1 | merges cleanly, small diff |
| #729 | Update rand requirement from 0.8 to 0.10 in /core/experimental/adamos_kernel | dependabot | 2026-02-25 | clean | 2 | 1 | merges cleanly, small diff |
| #1052 | Fix syntax error in swarm_runner.py causing CI failure | adamvangrover | 2026-05-13 | clean | 348 | 5 | merges cleanly, small diff |
| #1196 | Add Market Mayhem Vol 5 HTML Report | adamvangrover | 2026-07-27 | clean | 559 | 3 | merges cleanly, small diff |
| #1199 | feat(anomaly): implement LOF, One-Class SVM and ratio-based anomaly detection | adamvangrover | 2026-07-28 | clean | 66 | 1 | merges cleanly, small diff |
| #1417 | fix(ci): resolve Bandit B604 false positive and configure Ruff to extend pyproject.toml | adamvangrover | 2026-11-10 | clean | 4 | 2 | merges cleanly, small diff |
| #1433 | Bump react-dom and @types/react-dom in /services/webapp/client | dependabot | 2026-11-13 | clean | 127 | 2 | merges cleanly, small diff |

## P1 — Salvage / split

| PR | Title | Author | Created | Merge vs main | Lines | Files | Reason |
|---|---|---|---|---|---|---|---|
| #182 | Fix frontend build lint errors and resolve dependency backtracking | google-labs-jules | 2025-12-12 | conflict | 14 | 4 | conflicts with main; fix/security content worth salvaging |
| #183 | Stabilize Adam v23.0 Runtime: MockLLM, Dependency Fixes, and Orchestrator Patches | google-labs-jules | 2025-12-13 | conflict | 485 | 7 | conflicts with main; fix/security content worth salvaging |
| #261 | Enhancement: Documentation, Resilience, and Systematic Fixes | google-labs-jules | 2025-12-21 | conflict | 3638 | 100 | conflicts with main; fix/security content worth salvaging |
| #319 | Add Prompt Authoring Guide | google-labs-jules | 2026-01-04 | conflict | 189 | 1 | conflicts with main; fix/security content worth salvaging |
| #380 | Market Mayhem: Real Data Injection & UI Overhaul | google-labs-jules | 2026-01-14 | conflict | 52239 | 23 | conflicts with main; fix/security content worth salvaging |
| #475 | Fix CI Lint Errors | adamvangrover | 2026-02-02 | conflict | 749 | 40 | conflicts with main; fix/security content worth salvaging |
| #479 | Fix CI Lint Errors and Requirements | adamvangrover | 2026-02-02 | conflict | 55 | 9 | conflicts with main; fix/security content worth salvaging |
| #485 | Fix CI linting and dependency conflicts | adamvangrover | 2026-02-03 | conflict | 229 | 9 | conflicts with main; fix/security content worth salvaging |
| #491 | Fix backend-lint CI failures (missing imports) | adamvangrover | 2026-02-05 | conflict | 329 | 8 | conflicts with main; fix/security content worth salvaging |
| #493 | Fix flake8 undefined name errors | adamvangrover | 2026-02-05 | conflict | 765 | 15 | conflicts with main; fix/security content worth salvaging |
| #496 | Fix backend-lint CI failures by adding missing imports | adamvangrover | 2026-02-05 | conflict | 8 | 6 | conflicts with main; fix/security content worth salvaging |
| #497 | Fix CI: Backend Dependencies and Frontend Build Errors | adamvangrover | 2026-02-05 | conflict | 45844 | 100 | conflicts with main; fix/security content worth salvaging |
| #502 | fix: Resolve undefined name linting errors in Core modules | adamvangrover | 2026-02-06 | conflict | 380 | 10 | conflicts with main; fix/security content worth salvaging |
| #512 | ⚡ Bolt: Fix backend linting errors | adamvangrover | 2026-02-07 | conflict | 360 | 17 | conflicts with main; fix/security content worth salvaging |
| #514 | Fix Python Dependencies: Numpy, Safe-Unpickler, YFinance | adamvangrover | 2026-02-07 | conflict | 1502 | 17 | conflicts with main; fix/security content worth salvaging |
| #518 | Fix CI Flake8 Failures | adamvangrover | 2026-02-07 | conflict | 139 | 29 | conflicts with main; fix/security content worth salvaging |
| #536 | Fix CI Dependency Conflicts & Frontend Linting | adamvangrover | 2026-02-09 | conflict | 305 | 11 | conflicts with main; fix/security content worth salvaging |
| #559 | Fix Market Mayhem Archive HTML Generation | adamvangrover | 2026-02-11 | conflict | 460 | 3 | conflicts with main; fix/security content worth salvaging |
| #587 | fix: Showcase HTML & Asset Resolution | adamvangrover | 2026-02-13 | conflict | 5936 | 27 | conflicts with main; fix/security content worth salvaging |
| #746 | Fix CI Dependencies and Security Scan Issues | adamvangrover | 2026-02-25 | conflict | 1043 | 44 | conflicts with main; fix/security content worth salvaging |
| #756 | Add Market Mayhem Archive v23.5 Dashboard | adamvangrover | 2026-02-27 | clean | 823 | 1 | merges cleanly but 823 changed lines — split |
| #780 | Fix loading issues in showcase HTML files | adamvangrover | 2026-03-01 | conflict | 61319 | 16 | conflicts with main; fix/security content worth salvaging |
| #783 | Fix SQL injection vulnerability in query_sql tools and CI findings | adamvangrover | 2026-03-02 | conflict | 116 | 31 | conflicts with main; fix/security content worth salvaging |
| #785 | Fix showcase UI loading errors, script duplication, and missing data files | adamvangrover | 2026-03-02 | conflict | 386 | 39 | conflicts with main; fix/security content worth salvaging |
| #789 | fix: import asyncio in geopolitical_risk_agent.py | adamvangrover | 2026-03-03 | conflict | 265 | 8 | conflicts with main; fix/security content worth salvaging |
| #794 | Fix truncated market pulse and deep dive text | adamvangrover | 2026-03-04 | conflict | 1432 | 8 | conflicts with main; fix/security content worth salvaging |
| #824 | fix(lint): restore missing List import in historical_loader causing CI failure | adamvangrover | 2026-03-08 | conflict | 953 | 18 | conflicts with main; fix/security content worth salvaging |
| #838 | 🛡️ Sentinel: [CRITICAL] Fix insecure deserialization in TechnicalAnalystAgent | adamvangrover | 2026-03-09 | conflict | 8 | 2 | conflicts with main; fix/security content worth salvaging |
| #858 | Fix Institutional Radar Database Test State Leakage | adamvangrover | 2026-03-12 | conflict | 157 | 10 | conflicts with main; fix/security content worth salvaging |
| #869 | 🛡️ Sentinel: [MEDIUM] Fix insecure temp dir and missing request timeout | adamvangrover | 2026-03-12 | conflict | 154 | 10 | conflicts with main; fix/security content worth salvaging |
| #877 | Fix Sovereign AI & Crypto Supercycle hallucinations | adamvangrover | 2026-03-14 | conflict | 431648 | 100 | conflicts with main; fix/security content worth salvaging |
| #885 | fix: mkdocs build failure in ci-docs.yml | adamvangrover | 2026-03-15 | conflict | 1913 | 14 | conflicts with main; fix/security content worth salvaging |
| #899 | ⚡ Bolt: Fix undefined name error in ReflectorAgent | adamvangrover | 2026-03-18 | conflict | 11 | 4 | conflicts with main; fix/security content worth salvaging |
| #907 | Fix ReflectorAgent CI NameError | adamvangrover | 2026-03-19 | conflict | 562 | 6 | conflicts with main; fix/security content worth salvaging |
| #921 | Fix CI Pipeline: Universal Ingestor Deadlocks & Missing ML Packages | adamvangrover | 2026-03-24 | conflict | 184 | 10 | conflicts with main; fix/security content worth salvaging |
| #924 | Fix: resolve undefined name 'context' in reflector_agent | adamvangrover | 2026-03-24 | conflict | 361 | 9 | conflicts with main; fix/security content worth salvaging |
| #939 | Refactor: Merge & Purge of Legacy v23 Graph Engine | adamvangrover | 2026-03-30 | clean | 1229 | 34 | merges cleanly but 1229 changed lines — split |
| #973 | Fix GitHub CI failure: Missing AsyncMock import | adamvangrover | 2026-04-07 | conflict | 2634 | 24 | conflicts with main; fix/security content worth salvaging |
| #974 | Fix CI test failure by importing AsyncMock in test_server_governance.py | adamvangrover | 2026-04-08 | conflict | 521 | 9 | conflicts with main; fix/security content worth salvaging |
| #978 | fix(ci): Add missing AsyncMock import in test_server_governance.py to resolve F821 linting | adamvangrover | 2026-04-08 | conflict | 911 | 8 | conflicts with main; fix/security content worth salvaging |
| #979 | Fix missing AsyncMock import causing CI failure | adamvangrover | 2026-04-08 | conflict | 574 | 9 | conflicts with main; fix/security content worth salvaging |
| #1015 | fix: resolve unterminated triple-quoted string literal in base_agent.py | adamvangrover | 2026-04-21 | conflict | 9 | 2 | conflicts with main; fix/security content worth salvaging |
| #1018 | feat: Optimizer Performance and Security Hardening | adamvangrover | 2026-04-21 | conflict | 1406 | 53 | conflicts with main; fix/security content worth salvaging |
| #1046 | 🛡️ Sentinel: [MEDIUM] Fix missing timeout in external API call & fix syntax error | adamvangrover | 2026-05-11 | conflict | 9 | 3 | conflicts with main; fix/security content worth salvaging |
| #1075 | Fix bandit B615 warning in core/ppo_trainer.py | adamvangrover | 2026-05-22 | conflict | 65 | 4 | conflicts with main; fix/security content worth salvaging |
| #1081 | fix: Resolve CI failures in evaluator and ppo_trainer | adamvangrover | 2026-05-23 | conflict | 164 | 4 | conflicts with main; fix/security content worth salvaging |
| #1103 | fix: eliminate spaces and special characters from static files for github pages | adamvangrover | 2026-05-29 | conflict | 1048 | 72 | conflicts with main; fix/security content worth salvaging |
| #1137 | chore(deps): fix CI pip dependency resolution conflict | adamvangrover | 2026-06-18 | conflict | 324 | 3 | conflicts with main; fix/security content worth salvaging |
| #1138 | fix: resolve CI dependency conflicts and remove redundant deps | adamvangrover | 2026-06-18 | conflict | 290 | 5 | conflicts with main; fix/security content worth salvaging |
| #1141 | Fix CI pip dependency resolution conflicts | adamvangrover | 2026-06-20 | conflict | 572 | 2 | conflicts with main; fix/security content worth salvaging |
| #1148 | fix(deps): relax exact version bounds in requirements/base.txt to prevent CI resolution co | adamvangrover | 2026-06-21 | conflict | 174 | 3 | conflicts with main; fix/security content worth salvaging |
| #1234 | Fix scenario name mismatch in crisis simulation evaluation | adamvangrover | 2026-08-10 | conflict | 6 | 2 | conflicts with main; fix/security content worth salvaging |
| #1251 | fix: Resolve CI dependency timeout and implement Adam OS vNext architecture | adamvangrover | 2026-08-17 | conflict | 786 | 39 | conflicts with main; fix/security content worth salvaging |
| #1252 | fix: Resolve CI build failures and restore uv.lock integrity | adamvangrover | 2026-08-17 | conflict | 9190 | 25 | conflicts with main; fix/security content worth salvaging |
| #1266 | fix: resolve bandit security failures in CI pipeline | adamvangrover | 2026-08-22 | conflict | 1693 | 20 | conflicts with main; fix/security content worth salvaging |
| #1269 | fix: fix maturin build error and cargo config | adamvangrover | 2026-08-22 | conflict | 7590 | 12 | conflicts with main; fix/security content worth salvaging |
| #1290 | Enhance Digital Twin Model in Adam Research | adamvangrover | 2026-09-05 | conflict | 1363 | 2 | conflicts with main; recent — rebase |
| #1310 | fix: resolve module and dependency errors for testing | adamvangrover | 2026-09-18 | conflict | 9948 | 23 | conflicts with main; fix/security content worth salvaging |
| #1318 | fix(ci): Resolve Bandit security warnings in data scripts | adamvangrover | 2026-09-24 | conflict | 524 | 16 | conflicts with main; fix/security content worth salvaging |
| #1330 | Expand Market Mayhem Oct 02, 2026 Synthesis with Trailing Time-Series & YoY Comparative Me | adamvangrover | 2026-10-03 | conflict | 435 | 3 | conflicts with main; recent — rebase |
| #1334 | feat: additive pyproject configuration & pre-execution AST code evaluation engine | adamvangrover | 2026-10-07 | conflict | 5802 | 9 | conflicts with main; recent — rebase |
| #1335 | feat: Add ADAM standalone distribution page and update omniIndex and … | adamvangrover | 2026-10-07 | conflict | 5061 | 2 | conflicts with main; recent — rebase |
| #1340 | Fix pyproject.toml build system and lockfile alignment | adamvangrover | 2026-10-10 | conflict | 6956 | 1 | conflicts with main; fix/security content worth salvaging |
| #1341 | docs: rebuild 4 core markdown specification files | adamvangrover | 2026-10-10 | conflict | 752 | 4 | conflicts with main; recent — rebase |
| #1343 | Implement EVE v2.0 Institutional Single-Name Underwriting Engine | adamvangrover | 2026-10-10 | conflict | 6176 | 6 | conflicts with main; recent — rebase |
| #1344 | feat(finance): implement EVE v2.0 underwriting engine and master prompt | adamvangrover | 2026-10-10 | clean | 97438 | 6 | merges cleanly but 97438 changed lines — split |
| #1352 | Add ADAM Credit Swarm Orchestrator and Deterministic Judge Kernel | adamvangrover | 2026-10-17 | conflict | 653 | 3 | conflicts with main; recent — rebase |
| #1354 | Fix Dynamic Import Security Risks | adamvangrover | 2026-10-17 | conflict | 84 | 3 | conflicts with main; fix/security content worth salvaging |
| #1355 | Align pyproject.toml packaging, extras, and README quickstart | adamvangrover | 2026-10-17 | conflict | 469 | 3 | conflicts with main; recent — rebase |
| #1356 | Implement EVE v2.0 Institutional Single-Name Underwriting Architecture | adamvangrover | 2026-10-17 | conflict | 1412 | 5 | conflicts with main; recent — rebase |
| #1358 | Clean-Room LLM-as-a-Judge Evaluator and Context Memory Hub Upgrade | adamvangrover | 2026-10-18 | conflict | 615 | 4 | conflicts with main; recent — rebase |
| #1369 | Fix CI Failures in Bandit Security Check and Ruff Linter for Harness² | adamvangrover | 2026-10-22 | conflict | 959 | 43 | conflicts with main; fix/security content worth salvaging |
| #1372 | ADAM OS v30.1 — CI Fix & Continuous Architecture Replay | adamvangrover | 2026-10-22 | conflict | 405 | 13 | conflicts with main; fix/security content worth salvaging |
| #1379 | Fix Ruff linting errors for AFOS v30.1-R2-Expanded Orchestrator | adamvangrover | 2026-10-24 | conflict | 1063 | 10 | conflicts with main; fix/security content worth salvaging |
| #1381 | Implement ADAM October 2026 Frontier Upgrade Remediation & Resolve CI Gates | adamvangrover | 2026-10-26 | conflict | 1372 | 15 | conflicts with main; recent — rebase |
| #1399 | fix(tests): resolve ruff linting violations and type annotations in test files | adamvangrover | 2026-10-30 | conflict | 133 | 8 | conflicts with main; fix/security content worth salvaging |
| #1403 | Add Antigravity SDK, JulesAgent Replacement, and CI Hygiene Config | adamvangrover | 2026-10-30 | conflict | 871 | 12 | conflicts with main; recent — rebase |
| #1406 | feat: generate daily market brief and fix test typing | adamvangrover | 2026-11-01 | conflict | 423 | 14 | conflicts with main; fix/security content worth salvaging |
| #1409 | fix(ci): restore redundant hash check and configure global ruff exceptions | adamvangrover | 2026-11-01 | conflict | 165 | 5 | conflicts with main; fix/security content worth salvaging |
| #1414 | 1990s Web Browser Neural Network Repository Index | adamvangrover | 2026-11-07 | conflict | 3072 | 6 | conflicts with main; recent — rebase |
| #1415 | Bump the npm_and_yarn group across 1 directory with 13 updates | dependabot | 2026-11-07 | clean | 1908 | 3 | merges cleanly but 1908 changed lines — split |
| #1418 | Fix CI Ruff configuration and formalize AFOS Market Mayhem Nov 7, 2026 | adamvangrover | 2026-11-10 | conflict | 1358 | 12 | conflicts with main; fix/security content worth salvaging |
| #1420 | ADAM v30.1 Minimal-Additive Spec & Build (ADAM-PORT-CORE) | adamvangrover | 2026-11-10 | conflict | 1028 | 15 | conflicts with main; recent — rebase |
| #1422 | feat(newsletter): add Nov 7 bitemporal macro brief and update static navigation | adamvangrover | 2026-11-10 | conflict | 1363 | 9 | conflicts with main; recent — rebase |
| #1427 | Implement Antigravity Suite and ADAM v30.1 Minimal-Additive Spec (ADAM-PORT-CORE) | adamvangrover | 2026-11-11 | conflict | 1550 | 22 | conflicts with main; recent — rebase |

## P2 — Superseded / stale (close)

| PR | Title | Author | Created | Merge vs main | Lines | Files | Reason |
|---|---|---|---|---|---|---|---|
| #39 | Scaffold Adam v21.0 Project | google-labs-jules | 2025-11-13 | clean | 458 | 13 | 13/13 files target paths deleted from main |
| #49 | Scaffold v23 Planner and Self-Improvement Modules | google-labs-jules | 2025-11-21 | conflict | 1145 | 15 | 11/15 files target paths deleted from main |
| #70 | Adam v23.0 "Adaptive Hive" Implementation | google-labs-jules | 2025-11-29 | conflict | 970 | 7 | conflicts with main; stale (created before 2026-09) |
| #167 | Implement Adam v23.5 Showcase Generator | google-labs-jules | 2025-12-11 | conflict | 8611 | 69 | conflicts with main; stale (created before 2026-09) |
| #178 | Unified Adam v23.5 Platform Refactor | google-labs-jules | 2025-12-12 | conflict | 2288 | 14 | conflicts with main; stale (created before 2026-09) |
| #179 | Implement additional anomaly detection methods in AnomalyDetectionAgent | google-labs-jules | 2025-12-12 | conflict | 131 | 2 | conflicts with main; stale (created before 2026-09) |
| #205 | Enhance v23.5 Capabilities: Risk Models, MCP Logic, and Seed Data | google-labs-jules | 2025-12-16 | conflict | 1715 | 50 | 39/50 files target paths deleted from main |
| #237 | Enhance Showcase for Static GitHub Pages Deployment | google-labs-jules | 2025-12-19 | conflict | 423 | 7 | conflicts with main; stale (created before 2026-09) |
| #267 | Enhance Risk Core with Universal Data Fetcher and Counterfactual Logic | google-labs-jules | 2025-12-22 | conflict | 849 | 15 | conflicts with main; stale (created before 2026-09) |
| #296 | ⚡ Bolt: Optimized KG ingestion & Added Logging Dependency | google-labs-jules | 2025-12-29 | conflict | 127 | 4 | conflicts with main; stale (created before 2026-09) |
| #359 | Data-Driven Market Scenarios & Events | google-labs-jules | 2026-01-09 | conflict | 56819 | 50 | conflicts with main; stale (created before 2026-09) |
| #360 | Enhance Showcase UI with Interactive Logic | google-labs-jules | 2026-01-09 | conflict | 2353 | 15 | conflicts with main; stale (created before 2026-09) |
| #361 | Unified Showcase Portal & Repository Browser | google-labs-jules | 2026-01-10 | conflict | 10458 | 15 | conflicts with main; stale (created before 2026-09) |
| #381 | Gracefully merge market-mayhem-archive-16784488967544808505 | google-labs-jules | 2026-01-14 | conflict | 50525 | 26 | conflicts with main; stale (created before 2026-09) |
| #399 | Implement Adam Evolution Hub & Global Nav | google-labs-jules | 2026-01-16 | conflict | 29727 | 100 | conflicts with main; stale (created before 2026-09) |
| #415 | Enhance Showcase: Mission Control, Agent Registry, Prompt Library | google-labs-jules | 2026-01-18 | conflict | 18357 | 65 | conflicts with main; stale (created before 2026-09) |
| #546 | Integrate System Knowledge Graph into v24 Dashboard | adamvangrover | 2026-02-10 | conflict | 212158 | 8 | conflicts with main; stale (created before 2026-09) |
| #564 | Feature: Nexus Explorer & Workflow Visualizer | adamvangrover | 2026-02-12 | clean | 0 | 0 | no file changes |
| #565 | Enhance Nexus Simulation with live geopolitical bridge | adamvangrover | 2026-02-12 | clean | 0 | 0 | no file changes |
| #567 | Market Mayhem Archive 2.0: System Integration & Live Bridge | adamvangrover | 2026-02-12 | clean | 0 | 0 | no file changes |
| #569 | Enhance Nexus Simulation with Live Bridge and Geopolitical Events | adamvangrover | 2026-02-12 | clean | 0 | 0 | no file changes |
| #572 | ADAM OS v30.0 Expansion: Interactive Desktop Environment | adamvangrover | 2026-02-12 | clean | 0 | 0 | no file changes |
| #573 | Enhance Market Mayhem Archive with agent critiques and advanced visualizations | adamvangrover | 2026-02-12 | clean | 0 | 0 | no file changes |
| #579 | feat: Nexus Simulation interactive Monte Carlo engine | adamvangrover | 2026-02-13 | conflict | 898 | 6 | conflicts with main; stale (created before 2026-09) |
| #588 | Finalize Credit Memo Features: Interactive DCF & Historic Trends | adamvangrover | 2026-02-13 | conflict | 1659 | 13 | conflicts with main; stale (created before 2026-09) |
| #589 | Universal Data Loader for Credit Showcase | adamvangrover | 2026-02-13 | conflict | 2139 | 14 | conflicts with main; stale (created before 2026-09) |
| #590 | Advanced Credit Memo Features: Forecasts, Sensitivities & I/O | adamvangrover | 2026-02-13 | conflict | 2873 | 22 | conflicts with main; stale (created before 2026-09) |
| #601 | Align System to 2026 View | adamvangrover | 2026-02-14 | conflict | 60748 | 53 | conflicts with main; stale (created before 2026-09) |
| #604 | Credit Memo Automation 2026 Forecast & Facility Ratings | adamvangrover | 2026-02-14 | conflict | 57025 | 24 | conflicts with main; stale (created before 2026-09) |
| #611 | Enhance Market Mayhem: Cyberpunk Dashboard Transformation | adamvangrover | 2026-02-15 | conflict | 53978 | 70 | conflicts with main; stale (created before 2026-09) |
| #613 | Add Nexus Explorer and Workflow Visualizer | adamvangrover | 2026-02-15 | clean | 0 | 0 | no file changes |
| #616 | Enhance Market Mayhem: Interactive High Tier Reports & Navigation | adamvangrover | 2026-02-15 | conflict | 52901 | 100 | conflicts with main; stale (created before 2026-09) |
| #625 | Feat: Repo Intelligence & Network Map Enhancements | adamvangrover | 2026-02-16 | conflict | 4500 | 4 | conflicts with main; stale (created before 2026-09) |
| #639 | Add Market Mayhem Visual Repository | adamvangrover | 2026-02-16 | conflict | 2984 | 100 | conflicts with main; stale (created before 2026-09) |
| #650 | Market Mayhem: Crisis Simulator Module | adamvangrover | 2026-02-16 | conflict | 105692 | 100 | conflicts with main; stale (created before 2026-09) |
| #656 | Add Daily Briefing for Feb 11, 2026 | adamvangrover | 2026-02-17 | conflict | 906 | 5 | conflicts with main; stale (created before 2026-09) |
| #670 | Market Mayhem Archive: Enhanced Outlooks & UI | adamvangrover | 2026-02-18 | conflict | 21478 | 100 | conflicts with main; stale (created before 2026-09) |
| #681 | Update Credit Memo Automation with 2026 Forecasts & Repayment Schedules | adamvangrover | 2026-02-19 | conflict | 1983 | 24 | conflicts with main; stale (created before 2026-09) |
| #730 | Bump actions/checkout from 3 to 6 | dependabot | 2026-02-25 | conflict | 26 | 11 | conflicts with main; stale (created before 2026-09) |
| #731 | Bump actions/setup-node from 3 to 6 | dependabot | 2026-02-25 | conflict | 6 | 3 | conflicts with main; stale (created before 2026-09) |
| #732 | Bump actions/setup-python from 4 to 6 | dependabot | 2026-02-25 | conflict | 12 | 6 | conflicts with main; stale (created before 2026-09) |
| #733 | Bump actions/labeler from 5 to 6 | dependabot | 2026-02-25 | conflict | 2 | 1 | conflicts with main; stale (created before 2026-09) |
| #734 | Bump i18next-http-backend from 2.7.3 to 3.0.2 in /services/webapp/client | dependabot | 2026-02-25 | conflict | 196 | 3 | conflicts with main; stale (created before 2026-09) |
| #735 | Bump chart.js from 4.5.0 to 4.5.1 in /services/webapp/client | dependabot | 2026-02-25 | conflict | 188 | 3 | conflicts with main; stale (created before 2026-09) |
| #736 | Bump axios from 1.13.2 to 1.13.5 in /services/webapp/client | dependabot | 2026-02-25 | conflict | 200 | 3 | conflicts with main; stale (created before 2026-09) |
| #737 | Bump react-force-graph-2d from 1.29.0 to 1.29.1 in /services/webapp/client | dependabot | 2026-02-25 | conflict | 230 | 3 | conflicts with main; stale (created before 2026-09) |
| #738 | Bump i18next from 23.16.8 to 25.8.13 in /services/webapp/client | dependabot | 2026-02-25 | conflict | 239 | 3 | conflicts with main; stale (created before 2026-09) |
| #739 | Bump @testing-library/react from 16.3.0 to 16.3.2 in /services/webapp/client | dependabot | 2026-02-25 | conflict | 210 | 3 | conflicts with main; stale (created before 2026-09) |
| #740 | Bump @types/node from 24.10.2 to 25.3.0 in /services/webapp/client | dependabot | 2026-02-25 | conflict | 298 | 3 | conflicts with main; stale (created before 2026-09) |
| #741 | Bump web-vitals from 2.1.4 to 5.1.0 in /services/webapp/client | dependabot | 2026-02-25 | conflict | 196 | 3 | conflicts with main; stale (created before 2026-09) |
| #742 | Bump @testing-library/jest-dom from 6.8.0 to 6.9.1 in /services/webapp/client | dependabot | 2026-02-25 | conflict | 188 | 3 | conflicts with main; stale (created before 2026-09) |
| #747 | Credit Memo Pipeline Enhancement & Runtime Simulation | adamvangrover | 2026-02-25 | conflict | 5793 | 20 | conflicts with main; stale (created before 2026-09) |
| #751 | Market Mayhem Archive: Deep Analytics Upgrade | adamvangrover | 2026-02-26 | conflict | 8624 | 4 | conflicts with main; stale (created before 2026-09) |
| #755 | Enhance Report Libraries: Pagination, Filtering, and Engagement | adamvangrover | 2026-02-26 | conflict | 134 | 2 | conflicts with main; stale (created before 2026-09) |
| #761 | Enhance Credit Memo Pipeline with Valuation, Regulatory, and System 2 Critique | adamvangrover | 2026-02-27 | conflict | 4466 | 20 | conflicts with main; stale (created before 2026-09) |
| #763 | Market Mayhem Archive V2 Data Pipeline | adamvangrover | 2026-02-27 | clean | 0 | 0 | no file changes |
| #764 | Add AI Category Filter to Market Mayhem Archive | adamvangrover | 2026-02-27 | clean | 0 | 0 | no file changes |
| #765 | Add RealtimeFetcher for live financial data retrieval | adamvangrover | 2026-02-27 | clean | 0 | 0 | no file changes |
| #766 | Implement RealtimeFetcher for live financial data retrieval | adamvangrover | 2026-02-27 | clean | 0 | 0 | no file changes |
| #767 | Market Mayhem Archive V2: Dynamic UI & Advanced Charting | adamvangrover | 2026-02-27 | clean | 0 | 0 | no file changes |
| #768 | Fix Market Mayhem Archive V2 Integration | adamvangrover | 2026-02-27 | clean | 0 | 0 | no file changes |
| #769 | Enhance Market Mayhem Archive Dashboard with Auto-Refresh Script | adamvangrover | 2026-02-28 | conflict | 2805 | 28 | conflicts with main; stale (created before 2026-09) |
| #770 | Protocol ARCHITECT_INFINITE: CryptoArbitrageAgent Implementation | adamvangrover | 2026-02-28 | conflict | 587 | 29 | conflicts with main; stale (created before 2026-09) |
| #772 | feat: add Islamic Republic of Iran Geopolitical & Credit Risk Audit | adamvangrover | 2026-03-01 | clean | 0 | 0 | no file changes |
| #773 | Protocol ARCHITECT_INFINITE: Day 12 - OptionsFlowAgent Integration | adamvangrover | 2026-03-01 | clean | 0 | 0 | no file changes |
| #776 | 🛡️ Sentinel: [CRITICAL] Fix SQL injection in user query | adamvangrover | 2026-03-01 | clean | 0 | 0 | no file changes |
| #778 | Add RealtimeFetcher for live financial data | adamvangrover | 2026-03-01 | clean | 0 | 0 | no file changes |
| #792 | Verify ADAM-V-NEXT Components | adamvangrover | 2026-03-04 | clean | 0 | 0 | no file changes |
| #796 | Complete live data integration for all entities and format UI sections | adamvangrover | 2026-03-04 | conflict | 7114 | 39 | conflicts with main; stale (created before 2026-09) |
| #822 | Daily Ritual Robustness, Git Context, and Observability Improvements | adamvangrover | 2026-03-08 | conflict | 49 | 1 | conflicts with main; stale (created before 2026-09) |
| #832 | ⚡ Bolt: Address code review feedback on orphaned try/except blocks | adamvangrover | 2026-03-09 | conflict | 501 | 12 | conflicts with main; stale (created before 2026-09) |
| #835 | feat: apply ADAM-V-NEXT expansion markers to showcase and core components | adamvangrover | 2026-03-09 | conflict | 150 | 17 | conflicts with main; stale (created before 2026-09) |
| #837 | Enrich unified credit memos with deep financial data and expand dashboard | adamvangrover | 2026-03-09 | conflict | 260703 | 6 | conflicts with main; stale (created before 2026-09) |
| #845 | Add centralized terminal landing page | adamvangrover | 2026-03-10 | conflict | 1879 | 15 | conflicts with main; stale (created before 2026-09) |
| #848 | ⚡ Bolt: Vectorize detect_cluster_buys to improve performance | adamvangrover | 2026-03-11 | conflict | 12 | 2 | conflicts with main; stale (created before 2026-09) |
| #850 | Enhance Market Mayhem Archive V2 Dashboard Overlays | adamvangrover | 2026-03-11 | conflict | 8428 | 12 | conflicts with main; stale (created before 2026-09) |
| #851 | Update Black-Scholes and Bond Pricer to v2.2 | adamvangrover | 2026-03-11 | conflict | 30217 | 100 | conflicts with main; stale (created before 2026-09) |
| #852 | Create Artisanal Studio Dashboard | adamvangrover | 2026-03-11 | conflict | 399 | 4 | conflicts with main; stale (created before 2026-09) |
| #855 | feat: introduce DistressedDebtAgent for deep credit risk analysis | adamvangrover | 2026-03-12 | conflict | 110797 | 24 | conflicts with main; stale (created before 2026-09) |
| #859 | Standardize Legal, Macroeconomic, and SupplyChain Agents | adamvangrover | 2026-03-12 | conflict | 5308 | 15 | conflicts with main; stale (created before 2026-09) |
| #861 | docs: append architectural review to teacher_student.jsonl | adamvangrover | 2026-03-12 | conflict | 143 | 10 | conflicts with main; stale (created before 2026-09) |
| #862 | Refactor: Drastically prune codebase, remove dead weight, and consolidate utils | adamvangrover | 2026-03-12 | conflict | 42484 | 44 | conflicts with main; stale (created before 2026-09) |
| #863 | Add Analyst Consensus Data to Reporting Pipelines | adamvangrover | 2026-03-12 | conflict | 6773 | 29 | conflicts with main; stale (created before 2026-09) |
| #864 | Optimize and Prune Repository for LLM Context Windows | adamvangrover | 2026-03-12 | conflict | 19983 | 82 | conflicts with main; stale (created before 2026-09) |
| #871 | Refactor: 7-phase systematic upgrade of core showcase interface logic | adamvangrover | 2026-03-13 | conflict | 1303 | 100 | 91/100 files target paths deleted from main |
| #875 | docs: Add presentation outline for Agentic Automation Credit Risk | adamvangrover | 2026-03-14 | conflict | 162 | 15 | conflicts with main; stale (created before 2026-09) |
| #876 | feat: Market Mayhem Unified Dashboard | adamvangrover | 2026-03-14 | conflict | 8679 | 8 | conflicts with main; stale (created before 2026-09) |
| #879 | feat: Market Mayhem Archive Data and UI Enhancements | adamvangrover | 2026-03-14 | conflict | 34750 | 100 | conflicts with main; stale (created before 2026-09) |
| #886 | feat: Market Mayhem showcase UI upgrades with HITL review panel and visual components | adamvangrover | 2026-03-16 | conflict | 1662 | 100 | 95/100 files target paths deleted from main |
| #887 | Apply Master Upgrade Framework to Core UI Hub | adamvangrover | 2026-03-16 | conflict | 209793 | 100 | 94/100 files target paths deleted from main |
| #888 | Integrate new reports into index and market mayhem archive | adamvangrover | 2026-03-16 | conflict | 5568 | 100 | 94/100 files target paths deleted from main |
| #889 | feat: Enhance PydanticAgentBase with System 2 Sharpness and Swarm Integration | adamvangrover | 2026-03-16 | conflict | 257 | 100 | 98/100 files target paths deleted from main |
| #891 | Upgrade Market Mayhem Archive to 2.0 | adamvangrover | 2026-03-16 | conflict | 7717 | 100 | conflicts with main; stale (created before 2026-09) |
| #893 | ⚡ Bolt: Replace iterrows with to_dict for faster JSONL dataset generation | adamvangrover | 2026-03-16 | conflict | 146 | 100 | 99/100 files target paths deleted from main |
| #894 | Refine token fallback strategy to graceful compaction | adamvangrover | 2026-03-16 | conflict | 614 | 5 | conflicts with main; stale (created before 2026-09) |
| #895 | feat: Initialize LangGraph state machine for System 2 | adamvangrover | 2026-03-16 | conflict | 2879 | 100 | conflicts with main; stale (created before 2026-09) |
| #897 | feat(architecture): Implement Meta-Harness and Neuro-Symbolic System for ADAM v26.1 | adamvangrover | 2026-03-17 | conflict | 1793 | 100 | 95/100 files target paths deleted from main |
| #904 | fix(reflector): Resolve F821 undefined context and agent_schema import error | adamvangrover | 2026-03-19 | conflict | 134 | 100 | 97/100 files target paths deleted from main |
| #913 | feat(architecture): implement ADAM v26.1 meta-harness | adamvangrover | 2026-03-21 | conflict | 294267 | 21 | conflicts with main; stale (created before 2026-09) |
| #916 | feat: upgrade ingestion strategies to deep async pipeline with embeddings | adamvangrover | 2026-03-21 | conflict | 301392 | 37 | conflicts with main; stale (created before 2026-09) |
| #932 | ⚡ Bolt: Replace row-by-row df.apply with vectorized np.select in SEC 13F handler | adamvangrover | 2026-03-29 | conflict | 131 | 3 | conflicts with main; stale (created before 2026-09) |
| #935 | Architect: 7-Phase Upgrade of `data_utils.py` | adamvangrover | 2026-03-29 | conflict | 625 | 7 | conflicts with main; stale (created before 2026-09) |
| #953 | Add AI Credit Workflow Orchestration Report | adamvangrover | 2026-04-02 | conflict | 148 | 1 | conflicts with main; stale (created before 2026-09) |
| #969 | feat(showcase): Market Mayhem schemas, rubrics, and UI showcases | adamvangrover | 2026-04-07 | conflict | 1453 | 21 | conflicts with main; stale (created before 2026-09) |
| #984 | Add Risk Topography and System Setup modules to Nexus Portal | adamvangrover | 2026-04-09 | conflict | 112 | 5 | conflicts with main; stale (created before 2026-09) |
| #985 | feat: Update Adam Daily Brief for 2026-04-08 with full content | adamvangrover | 2026-04-09 | conflict | 1032 | 2 | conflicts with main; stale (created before 2026-09) |
| #993 | feat(jules): Protocol ARCHITECT_INFINITE implementation | adamvangrover | 2026-04-12 | conflict | 261 | 5 | conflicts with main; stale (created before 2026-09) |
| #1002 | feat: Add DynamicSearchAgent to v30 architecture | adamvangrover | 2026-04-15 | conflict | 164 | 4 | conflicts with main; stale (created before 2026-09) |
| #1017 | Resolve Phase 4 Architectural Reality Check bottlenecks | adamvangrover | 2026-04-21 | conflict | 424 | 10 | conflicts with main; stale (created before 2026-09) |
| #1022 | docs: comprehensive dependency consolidation and remediation plan | adamvangrover | 2026-04-23 | conflict | 4702 | 100 | 85/100 files target paths deleted from main |
| #1032 | Upgrade Status Terminal to Interactive Omni-Terminal | adamvangrover | 2026-04-30 | conflict | 13879 | 73 | conflicts with main; stale (created before 2026-09) |
| #1038 | feat: Comprehensive MacroEconomicAgent Analysis Enhancement | adamvangrover | 2026-05-02 | conflict | 300 | 14 | conflicts with main; stale (created before 2026-09) |
| #1139 | feat: Add Adam v3.0 Core Kernel Milestone 1 (kernel_rpc.json & architectural plan) | adamvangrover | 2026-06-18 | conflict | 279 | 5 | conflicts with main; stale (created before 2026-09) |
| #1153 | feat(jules): Additive Daily Recursive Cycle Legacy Workflow | adamvangrover | 2026-06-21 | conflict | 87 | 3 | conflicts with main; stale (created before 2026-09) |
| #1165 | feat: Professionalize architecture with PDIL JsonLogic, IaC, and Provenance | adamvangrover | 2026-06-26 | conflict | 1295 | 41 | conflicts with main; stale (created before 2026-09) |
| #1228 | State 0 Historical Data Ingestion | adamvangrover | 2026-08-06 | conflict | 1361 | 76 | conflicts with main; stale (created before 2026-09) |
| #1380 | ADAM Remediation Evaluation - Corrective Synthesis & Invariant Engine | adamvangrover | 2026-10-24 | conflict | 968 | 7 | 4/7 files target paths deleted from main |
| #1438 | Bump the cargo group across 6 directories with 5 updates | dependabot | 2026-11-21 | clean | 406 | 9 | bumps a frozen archive/snapshots lockfile |

## Regenerating

The inputs are `gh pr list --state open --json number,title,headRefName,createdAt,author,additions,deletions,files` plus the `git merge-tree` check above. Re-run after each merge wave; categories shift as `main` moves.
