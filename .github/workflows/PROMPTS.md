Here is the fully formatted and upgraded version of your ADAM Prompt Engineering Guide. I have enhanced the structure using industry-standard Markdown practices, improved scannability, and re-engineered the prompt templates themselves to align with current LLM research (specifically using the **Persona-Task-Context-Format** framework).

By structuring the prompts with clear constraints and section headers, you reduce AI hallucinations and guarantee highly deterministic outputs.

---

# 🚀 ADAM Prompt Engineering Guide: Complete Edition

To get the absolute best results from an AI assistant like ADAM for your repository, success comes down to providing clear, specific, and context-rich prompts. The secret to minimizing hallucinations and maximizing output quality is treating the AI as a specialized agent.

Industry research shows that the most effective prompts utilize the **Persona-Task-Context-Format** framework.

This guide is designed for both manual copy-pasting and automated machine workflows. Below are top-tier prompt templates, categorized by task, structurally optimized to maximize output quality.

---

## 🛠️ Code Refactoring & Improvement

These prompts help you clean up, optimize, and document your code at a micro level by forcing the AI to explain its reasoning.

### General Refactoring

```text
Act as a Senior Software Engineer specializing in [LANGUAGE/FRAMEWORK]. 

Task: Refactor the following code to improve its readability, performance, and adherence to best practices (e.g., DRY, SOLID). 
Constraint: Do NOT change its core functionality. 

Output Format:
1. The refactored code in a single code block.
2. A brief bullet-point list of the specific changes made and the reasoning behind them.

Code to Refactor:
```[PASTE YOUR CODE SNIPPET HERE]```

```

### Adding Comments & Docstrings

```text
Act as a Technical Writer and Software Engineer. 

Task: Generate clear, concise, and comprehensive docstrings for the following [LANGUAGE] function/class.
Format: Use [SPECIFIC FORMAT, e.g., Google-style Python docstrings, JSDoc]. Ensure you explain the purpose, arguments, types, and return values clearly.

Code:
```[PASTE YOUR FUNCTION/CLASS HERE]```

```

### Writing Unit Tests

```text
Act as a Lead Quality Assurance Engineer. 

Task: Write comprehensive unit tests for the following [LANGUAGE] function using [TESTING FRAMEWORK, e.g., Jest, pytest]. 
Requirements: 
- Cover the primary success path (happy path).
- Cover at least two edge cases.
- Cover one error-handling/failure case.
- Utilize mock objects/stubs where necessary.

Code to Test:
```[PASTE YOUR FUNCTION HERE]```

```

---

## 🐛 Bug Fixing & Troubleshooting

When things break, use these prompts to force the AI into an analytical state, identifying root causes before deploying safe, tested fixes.

### Root Cause Analysis & Fix

```text
Act as a Senior Debugging Expert and Systems Architect. I am encountering an error in my [LANGUAGE/FRAMEWORK] application.

Task: 
1. Identify the root cause of the error.
2. Explain clearly and concisely why it is happening.
3. Provide the exact code changes required to fix it safely.

Error/Stack Trace:
```[PASTE STACK TRACE HERE]```

Relevant Code:
```[PASTE CODE HERE]```

```

### Edge Case Discovery

```text
Act as an aggressive QA Tester and Security Researcher. 

Context: Assume the following code block functions correctly for standard inputs.
Task: 
1. Identify 3 potential edge cases, malicious inputs, or states that could cause this code to fail, panic, or act unexpectedly. 
2. Write the updated code to safely handle these edge cases.

Code to Review:
```[PASTE YOUR CODE HERE]```

```

---

## 👀 Code Review & Quality Assurance

Use these prompts to have the AI act as a strict, senior reviewer before you merge a Pull Request.

### Strict Security & Performance Review

```text
Act as a Principal Staff Engineer and Application Security Auditor. 

Task: Perform a comprehensive code review on the following snippet.
Focus Areas:
- Security vulnerabilities (e.g., OWASP Top 10, injection flaws, data leaks).
- Performance bottlenecks (e.g., Big O time/space complexity issues, memory leaks).
- Architectural anti-patterns.

Format: Present your response as a formal Pull Request review with actionable, constructive comments and inline code suggestions.

Code to Review:
```[PASTE YOUR CODE HERE]```

```

---

## 🔀 Version Control: Commits, Pushes, and Pulls

Clear version control history and smooth branch management are vital. These prompts guide safe Git operations and standard-compliant documentation.

### Generating a Commit Message from a Diff

```text
Act as an Expert Developer enforcing the Conventional Commits specification. 

Task: Based on the following git diff, write a perfect commit message. 
Format Requirements:
- A structural header: <type>(<scope>): <short description>
- A detailed body explaining the 'what' and 'why' of the changes (skip the 'how' unless complex).
- Do not use markdown formatting in the final output, just raw text ready for the terminal.

Git Diff:
```[PASTE YOUR GIT DIFF OUTPUT HERE]```

```

### Safe Pull & Push Conflict Resolution Guide

```text
Act as a Git Workflow Expert. 

Context: I am currently on branch `[MY-FEATURE-BRANCH]`. I need to pull changes from `[ORIGIN/MAIN]`, but I have local uncommitted changes and I suspect there will be merge conflicts. 
Task: Provide the exact, step-by-step sequence of terminal commands to:
1. Safely stash my changes.
2. Pull the main branch.
3. Reapply my stash.
4. Resolve conflicts.
5. Commit and push my branch securely.

Include brief inline comments explaining what each command does.

```

---

## 🤖 GitHub Copilot Optimization

Sometimes you need ADAM to prepare your codebase so that inline AI tools (like Copilot or Cursor) work flawlessly.

### Generating Copilot-Friendly Scaffolding

```text
Act as a Code Architect. 

Task: I need to implement [DESCRIBE THE COMPLEX BUSINESS LOGIC/ALGORITHM]. 
Constraint: Do NOT write the actual functional code. 
Action: Write a series of highly specific, sequential, and descriptive inline comments formatted for [LANGUAGE]. 

Goal: Make these comments so explicit and structured that an inline AI like GitHub Copilot can perfectly predict and generate the correct implementation step-by-step beneath each comment block.

```

### Prompting for Type Definitions to Guide Copilot

```text
Act as a Domain-Driven Design Expert. 

Context: I am starting a new feature in [TypeScript/Python]. 
Task: Before I write the logic, generate comprehensive, strictly-typed Interfaces/Types/Pydantic Models for a [DESCRIBE THE DOMAIN, e.g., E-commerce Checkout System]. 
Requirements: Include descriptive JSDoc/docstrings on every single property. 

Goal: I will use these definitions to ground GitHub Copilot's context for the rest of the application.

```

---

## 📝 Documentation (READMEs & More)

Good documentation is the lifeblood of a maintainable repository. These prompts format the AI's output specifically for standard Markdown documents.

### Generating a README from Scratch

```text
Act as a Technical Writer and Open Source Maintainer. Generate a professional README.md file for my project.

Project Context:
- Project Name: [YOUR PROJECT NAME]
- Description: [ONE-SENTENCE SUMMARY OF WHAT IT DOES]
- Key Features: [LIST 3-5 KEY FEATURES]
- Tech Stack: [LIST MAIN LANGUAGES/FRAMEWORKS]

Structure Requirements: Include clean Markdown headers for Introduction, Features, Installation, Usage, and Contributing. Use code blocks for any terminal commands.

```

### Creating a CONTRIBUTING.md

```text
Act as a Community Manager for an open-source project. 

Task: Create a CONTRIBUTING.md file. 
Tone: Welcoming, inclusive, and encouraging. 
Required Sections:
1. How to Report Bugs
2. How to Suggest Features
3. How to Set Up the Development Environment
4. Pull Request Process (Note: We use Conventional Commits and require unit tests for all new features).

```

---

## ⚙️ CI/CD Automation & Workflow Upgrades

Ensure security, speed up builds, and align workflows with your repository's evolving architecture.

### Bulk Workflow Modernization

```text
Act as a Lead DevOps Engineer. 

Task: Review my current GitHub Actions workflow below. 
1. Upgrade all deprecated actions to their latest major versions.
2. Update runtime environments (e.g., Python, Node) to modern standards.
3. Implement dependency caching to speed up the build time.

Output: Provide the clean, updated YAML file in a single code block, followed by a brief summary of what was modernized.

Current Workflow:
```[PASTE YOUR WORKFLOW.YML HERE]```

```

### Continuous Security & Compliance

```text
Act as a DevSecOps Engineer. 

Task: Enhance my existing CI pipeline by integrating continuous security checks. 
Requirements:
- Add dedicated jobs to run dependency vulnerability scans (e.g., npm audit, pip-audit, or cargo audit).
- Add SAST tools (e.g., Bandit, CodeQL).
- Ensure the workflow fails if high-severity vulnerabilities are found.

Output: Provide the updated YAML.

```

---

## 🔬 Auto-Research & Technology Scouting

Have the AI perform heavy lifting on architectural decisions and library comparisons before you write a single line of code.

### Comparative Technology Analysis

```text
Act as a Lead Solutions Architect. 

Task: Conduct a comparative research analysis on the top 3 open-source libraries for [SPECIFIC TASK, e.g., vector database search] in [LANGUAGE]. 
Evaluation Criteria: Performance, community support/activity, ease of integration, and licensing. 

Output Format: 
1. A Markdown table comparing the features side-by-side.
2. A final, highly-justified recommendation for a high-traffic production environment.

```

---

## 🧠 Machine Learning & Data Science

Prompts designed specifically for AI/ML repositories, data pipelines, and model training.

### Generating a Complete Training Pipeline

```text
Act as an Expert Machine Learning Engineer. 

Task: Generate a complete, modular, and reproducible training pipeline in [PyTorch/TensorFlow] for [SPECIFIC TASK, e.g., tabular data classification]. 
Requirements:
- Data loading and preprocessing logic.
- Neural network model definition.
- A robust training loop with validation.
- Early stopping based on validation loss.
- Logic to save the best model weights.

Constraint: Add rich, educational comments explaining the hyperparameter choices and tensor shape transformations.

```

### Automated Data Cleaning Script

```text
Act as a Senior Data Scientist. 

Task: Write a Python script using `pandas` to automate the cleaning of a messy dataset. 
Requirements:
1. Handle missing values (impute numeric, drop/fill categorical).
2. Remove duplicate rows.
3. Detect and handle outliers in continuous variables using the IQR method.
4. Normalize all column names to standard `snake_case`.

Design: Ensure the code is wrapped in a reusable, well-documented class or function.

```

---

## 🏗️ Repository Strategy & Structure

Use these prompts for higher-level architectural planning and structural organization.

### Suggesting a Directory Structure

```text
Act as a Software Architect. 

Context: I am starting a new [TYPE OF PROJECT, e.g., Node.js/Express API, React web app, Python data science project]. 
Task: Suggest a clean, scalable, and industry-standard directory structure. 

Output Format: 
1. Provide the structure as an ASCII tree in a code block.
2. Briefly explain the architectural purpose of each top-level directory.

```

---

> **💡 Pro Tip:** By using these targeted, variable-based prompts, you establish a strictly deterministic logic flow. Whether you are manually chatting with an AI or injecting these prompts programmatically via API pipelines, this structural consistency guarantees significantly more useful, accurate, and production-ready outputs.
