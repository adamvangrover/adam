# AFOS Policy Kernel Specification

## Overview
The Policy Kernel is the deterministic engine of AFOS. It elevates business logic, financial covenants, and regulatory requirements from hardcoded application logic into independently versioned, testable, and computable assets.

## The Policy Compilation Pipeline

1. **Policy DSL (Domain Specific Language):** Policies are authored in human-readable or business-analyst-friendly formats.
2. **Parser:** Translates the DSL into an intermediate representation.
3. **AST (Abstract Syntax Tree):** Structurally maps the logic.
4. **Compiler:** Converts the AST into the target execution format.
5. **Optimization:** Simplifies boolean logic and caches constants.
6. **Execution DAG:** Defines the dependency order for evaluation.
7. **Deterministic Runtime:** Evaluates the DAG against a given state.

## Supported Runtimes
The compiler is designed to eventually support execution across multiple targets through the exact same engine:
* **JsonLogic:** For lightweight, stateless boolean and mathematical evaluations.
* **DMN (Decision Model and Notation):** For complex decision tables.
* **SQL Predicates:** For pushing logic down to the Transactional Memory tier.
* **YAML Policies:** For infrastructure and access control (e.g., OPA/Cedar).

## Core Principles
* **Stateless Execution:** Policy evaluation must not produce side effects.
* **Determinism:** Given the same policy version and the same input state, the evaluation must always yield the exact same result.
* **Independent Testability:** Policies must be testable in isolation from workflows or data ingestion pipelines.
