🚀 ADAM Prompt Engineering Guide: Complete EditionTo get the best results from an AI assistant like ADAM for your repository, it is all about providing clear, specific, and context-rich prompts. The key is to treat the AI as a specialized assistant by assigning it a role, context, and a clear goal.This guide is designed for both manual copy-pasting and automated machine workflows. Below are top-tier prompt templates categorized by task to maximize your output quality.🛠️ Code Refactoring & ImprovementThese prompts help you clean up, optimize, and document your code at a micro level.General RefactoringAct as a senior software engineer specializing in <language, e.g., Python>. Refactor the following code to improve its readability, performance, and adherence to best practices like <DRY, SOLID>. Do not change its functionality. Provide the refactored code and a brief bullet-point list of the key changes you made.

[Paste your code snippet here]
Adding Comments & DocstringsGenerate clear and concise docstrings for the following <language> function/class in <a specific format, e.g., Google-style Python docstrings, JSDoc>. Explain the purpose, arguments, and return value.

[Paste your function or class here]
Writing Unit TestsAct as a Quality Assurance engineer. Write unit tests for the following <language> function using the <testing framework, e.g., Jest, pytest>. Cover the primary success case, at least two edge cases, and one error-handling case. Include mock objects if necessary.

[Paste your function here]
🐛 Bug Fixing & TroubleshootingWhen things break, use these prompts to identify root causes and deploy safe, tested fixes.Root Cause Analysis & FixAct as a Senior Debugging Expert. I am getting the following error in my <language/framework> application. 

Error/Stack Trace:
[Paste stack trace here]

Relevant Code:
[Paste code here]

Identify the root cause, explain clearly why it is happening, and provide the exact code changes required to fix it safely.
Edge Case DiscoveryReview the following code block. Assume it functions correctly for standard inputs. Act as a malicious user or edge-case tester. Identify 3 potential edge cases or inputs that could cause this code to fail, panic, or act unexpectedly. Then, write the code to handle these edge cases.

[Paste your code here]
👀 Code Review & Quality AssuranceUse these prompts to have the AI act as an extra set of eyes before you merge a Pull Request.Strict Security & Performance ReviewAct as a Principal Staff Engineer and Security Auditor. Perform a comprehensive code review on the following snippet. 
1. Flag any security vulnerabilities (e.g., OWASP top 10, injection flaws, data leaks).
2. Identify performance bottlenecks (e.g., Big O time/space complexity issues).
3. Point out any anti-patterns.
Format your response as a formal Pull Request review with actionable, constructive comments and code suggestions.

[Paste your code here]
🔀 Version Control: Commits, Pushes, and PullsClear version control history and smooth branch management are vital. These prompts guide safe Git operations and documentation.Generating a Commit Message from a DiffAct as an expert developer who follows the Conventional Commits specification. Based on the following git diff, write a perfect commit message. The message should have a type, a scope (optional), a short description, and a more detailed body explaining the 'what' and 'why'.

[Paste your git diff output here]
Safe Pull & Push Conflict Resolution GuideAct as a Git workflow expert. I am currently on branch `<my-feature-branch>`. I need to pull changes from `<origin/main>`, but I have local uncommitted changes and I suspect there will be merge conflicts. Provide the exact, step-by-step sequence of terminal commands to safely stash my changes, pull the main branch, reapply my stash, resolve conflicts, commit, and push my branch securely.
🤖 GitHub Copilot OptimizationSometimes you need ADAM to prepare your codebase so that inline AI (like GitHub Copilot) works flawlessly.Generating Copilot-Friendly ScaffoldingAct as a Code Architect. I need to implement <describe the complex business logic/algorithm>. Do NOT write the actual code. Instead, write a series of highly specific, sequential, and descriptive inline comments formatted for <language>. The goal is to make these comments so explicit and structured that GitHub Copilot can perfectly predict and generate the correct implementation beneath each comment block.
Prompting for Type Definitions to Guide CopilotI am starting a new feature in <TypeScript/Python>. Before I write the logic, generate comprehensive, strictly-typed Interfaces/Types/Pydantic Models for a <describe the domain, e.g., E-commerce Checkout System>. Include descriptive JSDoc/docstrings on every property. I will use these definitions to ground GitHub Copilot's context.
📝 Documentation (READMEs & More)Good documentation is crucial for any repository. These prompts save you hours of writing and formatting.Generating a README from ScratchGenerate a professional README.md file for my project.

Project Name: <Your Project Name>
Description: <A one-sentence summary of what your project does>
Key Features: <List 3-5 key features as bullet points>
Tech Stack: <List the main languages, frameworks, and tools>

Include sections for: Introduction, Features, Installation, Usage, and Contributing. Format with clean Markdown headers and code blocks.
Creating a CONTRIBUTING.mdCreate a CONTRIBUTING.md file for my open-source project. The tone should be welcoming and encouraging. Include sections on:
- How to Report Bugs
- How to Suggest Features
- How to Set Up the Development Environment
- Pull Request Process (mention that we use Conventional Commits and require tests for new features).
⚙️ CI/CD Automation & Workflow UpgradesEnsure security, speed up builds, and align workflows with your repository's evolving architecture.Bulk Workflow ModernizationAct as a Lead DevOps Engineer. Review my current GitHub Actions workflow below. Please upgrade all deprecated actions to their latest major versions, update runtime environments (e.g., Python, Node) to modern standards, and implement dependency caching to speed up the build. Provide the clean, updated YAML file.

[Paste your workflow.yml here]
Continuous Security & ComplianceEnhance my existing CI pipeline by integrating continuous security checks. Add dedicated jobs to run dependency vulnerability scans (e.g., npm audit, pip-audit, cargo audit) and SAST tools (e.g., Bandit, CodeQL). Ensure the build fails if high-severity vulnerabilities are found. Provide the updated YAML.
🔬 Auto-Research & Technology ScoutingHave the AI perform heavy lifting on architectural decisions and library comparisons before you write a single line of code.Comparative Technology AnalysisAct as a Lead Solutions Architect. Conduct a comparative research analysis on the top 3 open-source libraries for <specific task, e.g., vector database search> in <language>. Compare them based on performance, community support, ease of integration, and licensing. Output a Markdown table comparing the features, followed by a final, justified recommendation for a production environment.
🧠 Machine Learning & Data SciencePrompts designed specifically for AI/ML repositories, data pipelines, and model training.Generating a Complete Training PipelineAct as an expert Machine Learning Engineer. Generate a complete, modular, and reproducible training pipeline in <PyTorch/TensorFlow> for <specific task, e.g., tabular data classification>. Include:
1. Data loading and preprocessing.
2. The neural network model definition.
3. A robust training loop with validation.
4. Early stopping based on validation loss.
5. Code to save the best model weights.
Add rich comments explaining the hyperparameter choices.
Automated Data Cleaning ScriptAct as a Senior Data Scientist. Write a Python script using pandas to automate the cleaning of a messy dataset. The script should:
1. Handle missing values (impute numeric, drop or fill categorical).
2. Remove duplicate rows.
3. Detect and handle outliers in continuous variables using the IQR method.
4. Normalize the column names (snake_case).
Ensure the code is wrapped in a reusable class or function.
🏗️ Repository Strategy & StructureUse these prompts for higher-level architectural planning and structural organization.Suggesting a Directory StructureI am starting a new <type of project, e.g., Node.js/Express API, React web app, Python data science project>. Suggest a clean, scalable, and conventional directory structure. Provide the structure as an ASCII tree and briefly explain the purpose of each top-level directory.
💡 Pro Tip: By using these targeted and context-aware prompts, you establish a more deterministic logic flow. Whether you are manually chatting with an AI or injecting these prompts programmatically via API, this structure guarantees significantly more useful, accurate, and repository-ready outputs.
