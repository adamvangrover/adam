You are an expert Software Architect and Senior Developer. I am systematically upgrading my entire codebase by feeding it to you in targeted, bite-sized chunks.

Your objective is to drastically refine, upgrade, and optimize the attached code by running it through a strict 7-phase enhancement cycle. Do not alter the core business logic or break existing integrations, but significantly improve how the code operates.

Please apply the following 7 phases sequentially to the provided code:

1. THE PRUNING (Waste Removal)
Aggressively but safely prune the code. Remove dead code, unused variables/imports, unneeded dependencies, and obsolete comments. Simplify overly complex or nested logic.

2. THE REFACTOR (Architecture & Modularity)
Reorganize the code to adhere to modern best practices. Improve modularity and readability. Abstract repetitive logic into reusable utility functions or classes, and apply structural design patterns that make it robust and scalable.

3. THE OPTIMIZER (Performance & Safety)
Audit for bottlenecks, resource leaks, and security vulnerabilities. Optimize for time and space complexity. Fortify error handling to ensure the application fails gracefully.

4. THE MODERNIZER (Upgrades & Syntax)
Upgrade the code to utilize the absolute latest language features and framework standards (e.g., modern ES6+, Python 3.10+ pattern matching). Replace outdated patterns and anti-patterns with contemporary idioms.

5. THE INNOVATOR (Advanced Capabilities)
Identify where this code could benefit from cutting-edge features (e.g., smart parsing, autonomous logic, or lightweight AI/LLM integrations). If applicable, seamlessly integrate the most high-value enhancement into the logic.

6. THE DOCUMENTER (Clarity & AI-Readability)
Generate clean, highly informative inline documentation and docstrings. Write them specifically so that future AI context windows can instantly understand the module's purpose, inputs, and outputs.

7. THE VALIDATOR (Test Generation)
Generate a comprehensive suite of unit and integration tests using a modern testing framework. Focus on core logic, edge cases, and error-handling. Mock external calls where necessary.

OUTPUT FORMAT:
Do not output the code 7 different times. Instead, process it internally and provide:
1. "Changelog:" A brief, bulleted summary of the specific improvements made during the 7 phases.
2. "Upgraded Code:" The final, consolidated, production-ready code.
3. "Test Suite:" The complete, ready-to-run testing file for the upgraded code.

Here is the code to process today:
[INSERT YOUR CODE/FILE HERE]

How to execute this systematically
If you want to automate this over time, here is how you structure your workflow:
 * Map the Repo: Generate a list of all your target files (e.g., find src -type f -name "*.py").
 * Prioritize: Start with foundational files (utilities, database connections, schemas) before moving to higher-level routing or UI components. You want the foundation upgraded before the things that rely on it are upgraded.
 * Pace Yourself: If doing this manually, do 1 to 3 files a day. Review the output, commit the changes, and test your app to ensure the LLM didn't hallucinate a breaking change.