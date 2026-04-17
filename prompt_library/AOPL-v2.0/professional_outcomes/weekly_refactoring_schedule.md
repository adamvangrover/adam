# Weekly Refactoring Schedule

This is a highly effective approach. By feeding your codebase to an LLM in targeted, bite-sized chunks with specific objectives, you can systematically upgrade your repository without overwhelming the context window.
Here is a weekly schedule of powerful, specialized prompts you can run daily against your code. Whenever you use these, paste the relevant prompt along with the specific files or modules you want to focus on that day.

## Monday: The Pruning (Waste & Dead Code Removal)
Goal: Strip out bloat, unused imports, and legacy logic to reduce context size and improve clarity.
The Prompt:
> "Analyze the attached codebase. Your task is to aggressively but safely prune this code. Identify and remove any dead code, unused variables, redundant functions, unneeded dependencies, and obsolete comments. Simplify overly complex or nested logic without altering the core behavior. Return the cleaned code and provide a bulleted summary of exactly what waste was removed and why."

## Tuesday: The Refactor (Architecture & Modularity)
Goal: Reorganize the code to be highly modular, making it easier for both humans and AI to read and expand upon.
The Prompt:
> "Review this code for adherence to modern software engineering best practices. Refactor the code to drastically improve modularity, readability, and maintainability. Abstract repetitive logic into reusable utility functions or classes. Suggest and implement structural changes or design patterns that make this codebase more robust and easier to scale. Return the refactored code."

## Wednesday: The Optimizer (Performance & Safety)
Goal: Ensure the code runs incredibly fast, handles resources properly, and is secure.
The Prompt:
> "Audit this code for performance bottlenecks, resource leaks, and potential security vulnerabilities. Optimize the logic for time and space complexity, ensuring efficient data handling and asynchronous operations where applicable. Fortify the error handling to ensure the application fails gracefully rather than crashing. Provide the optimized code and a brief explanation of the performance gains."

## Thursday: The Modernizer (Upgrades & Syntax)
Goal: Bring the code up to date with the absolute latest language features and framework standards.
The Prompt:
> "Examine this codebase and identify any areas that rely on outdated patterns, legacy syntax, or anti-patterns. Upgrade the code to utilize the absolute latest language features (e.g., modern ES6+ for JS, pattern matching in Python 3.10+, etc.). Ensure the code aligns with contemporary idioms for its language. Return the modernized code and note which new language features were implemented."

## Friday: The Innovator (Cutting-Edge AI Integration)
Goal: Build out functionality that leverages modern LLM capabilities, autonomous agents, or advanced APIs.
The Prompt:
> "Analyze the core purpose of this code. I want to upgrade this project by integrating cutting-edge AI capabilities (such as LLM calls, RAG pipelines, autonomous agent behaviors, or smart parsing). Brainstorm 3 specific, high-value AI features that could natively enhance this application. Then, write the production-ready implementation code for the best of those 3 ideas, ensuring it seamlessly integrates with my existing logic."

## Saturday: The Documenter (Clarity & Onboarding)
Goal: Ensure the refined code is perfectly documented so that future AI context windows can understand it instantly.
The Prompt:
> "Read through this updated code. Generate clean, concise, and highly informative inline documentation and docstrings for all major functions and classes. Then, write a comprehensive 'Architecture & Usage' section that I can append to my README. The documentation should be written in a way that helps another AI instantly understand the purpose, inputs, and outputs of this module."

## Sunday: The Validator (Test Generation)
Goal: Lock in your progress by ensuring everything works and won't break during next week's changes.
The Prompt:
> "Analyze this codebase and generate a comprehensive suite of unit and integration tests using a modern testing framework. Focus on testing the core logic, edge cases, and the error-handling mechanisms. Ensure the tests are highly readable and mock any external API calls or database connections. Return the complete test file."

Pro-Tip for Execution: When you run these, don't just dump the whole repo in at once. Pick a specific feature, folder, or file (e.g., "Today I am running the Monday and Tuesday prompts on my database utility functions") and cycle through your repository systematically.
