# 🚀 Async Coding Swarm Prompt: Improve adamvangrover/adam

**Context:**
*   **Repository:** `adamvangrover/adam`
*   **Description:** A complex AI-powered agent/assistant codebase containing core logic, agent definitions, prompt libraries, server components, and frontend.
*   **Goal:** Generate actionable code contributions, tests, docs, and quality fixes across the project.

## 🧠 Task Overview
We’re running an async coding swarm to break down improvements into discrete, parallel tasks. You can claim a task, comment progress, ask questions, and submit PRs.

## 🗂️ Priority Areas

### Project Health & CI/CD
*   🚧 Fix broken tests and flaky builds
*   🛠️ Add GitHub Actions workflows for lint, test, build, and deploy
*   📦 Ensure dependency pinning & reproducible environments

### Documentation
*   📘 Improve README with architecture diagram, module breakdown, getting started, and deployment guide
*   🧾 Generate API spec docs for core modules (server, agents)
*   🧪 Write CONTRIBUTING guide with coding standards

### Quality & Testing
*   🧪 Add unit tests for core logic
*   🧠 Add integration tests for agent workflows
*   📈 Add coverage reporting

### Feature Improvements
*   💬 Improve prompt library with categorization & examples
*   🤖 Add new agent behaviors (e.g., memory/session persistence)
*   🌐 Improve frontend UX/interaction flows

### Performance & Reliability
*   ⚙️ Profile and optimize slow tasks
*   🔁 Improve async task management (queue/retry logic)

## 📌 Claim Work Format
Post a comment in this thread/issue/discussion using this template:

```
🔹 I’m working on:

Area: <Choose from Health | Docs | Quality | Features | Performance>
Task:
Expected output: <Test/PR/template snippet/analysis>
ETA:
```

**Example:**
```
🔹 I’m working on:

Area: Documentation
Task: Add architecture diagram + component explainer in README
Expected output: README section + diagram asset
ETA: Jan 10
```

## 🧑‍💻 Task Prompts (Copy-Paste)

### 📌 Code Quality
**✅ Improve test coverage:**
*   Identify missing tests under `/core` and `/server`
*   Write `pytest`/`unittest` tests for examples and critical behaviors
*   Submit PR with coverage badge update and coverage ≥ 70%

### 📌 CI/CD Work
**🔁 Create/Enhance CI workflows:**
*   Add GitHub Action for: linting, type checking, testing, security scan
*   Ensure matrix build (Python versions, envs)
*   Add PR gating checks

### 📌 Documentation
**📘 Improve documentation:**
*   Add architecture overview (text + diagram)
*   List key modules: agents, core, data flow
*   Provide quickstart steps

### 📌 Feature: Prompt Lib
**🧠 Curate prompt library:**
*   Review existing prompts under `/prompt_library`
*   Categorize by purpose (QA, planning, analysis, debugging)
*   Add usage examples and expected outputs

### 📌 Agent Enhancements
**🤖 Extend agent behavior:**
*   Add memory persistence (e.g., store sessions to DB/Redis)
*   Write tests to validate memory recall use cases

## 📡 Async Communication Rules
*   Post blockers or design questions in the thread
*   Reference code locations (`path/to/file.py:line`) for clarity
*   Update status each 24–48 hours
*   Keep messages short and actionable

## 🧠 Optional AI Assistance
For each task, you can also request:
`/ai codegen`

**Provide:**
*   Code stub / patch
*   Short test +
*   Short explanation

*Only use for generating starting code, then refine manually.*
