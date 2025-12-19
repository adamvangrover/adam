# LIB-META-009: Async Coding Agent - Repo to Markdown CLI

*   **ID:** `LIB-META-009`
*   **Version:** `1.0`
*   **Author:** Adam v23.5
*   **Objective:** To instruct an asynchronous coding agent to build a specialized CLI microservice that downloads and serializes entire GitHub repositories into a single Markdown text file. This tool is essential for "grounding" LLMs in a codebase by converting the file structure and content into a prompt-friendly format.
*   **When to Use:** When you need to quickly ingest an external codebase or the current repository into an LLM context window. This prompt directs the creation of the tool itself.

---

### **Metadata & Configuration**

*   **Key Placeholders:**
    *   `[Target_Repo_URL]`: The default repository to download (e.g., `https://github.com/adamvangrover/adam`).
    *   `[Output_Directory]`: The subdirectory where the tool should be built (e.g., `tools/repo_to_markdown`).
    *   `[Max_Lines]`: The threshold for warnings or chunking (e.g., "100k").
*   **Pro-Tips for 'Adam' AI Integration:**
    *   **Agent:** `AsyncCodingAgent` or `DevOpsAgent`.
    *   **Discreet Subdirectory:** The tool should be self-contained in its own folder to avoid polluting the main repository.
    *   **Scalability:** The tool must handle 1k, 10k, and 100k line codebases gracefully, potentially by splitting outputs or ignoring lock files.

---

### **Example Usage**

```
[Target_Repo_URL]: "https://github.com/adamvangrover/adam"
[Output_Directory]: "tools/repo_ingestor"
```

---

## **Full Prompt Template**

```markdown
# ROLE: Asynchronous Python Developer & CLI Toolsmith

# CONTEXT:
You are an expert software engineer specializing in building high-performance CLI microservices. You excel at asynchronous I/O operations and file system manipulations. You are tasked with creating a standalone utility that acts as a "bridge" between raw code repositories and LLM context windows.

# GOAL:
Build a discreet, self-contained CLI microservice app in the directory **`[Output_Directory]`**.
The app's primary function is to download a GitHub repository (defaulting to **`[Target_Repo_URL]`**) and convert its entire contents into a single, formatted Markdown text file.

# SPECIFICATIONS:

### 1. **Project Structure**
Create the following structure within `[Output_Directory]`:
*   `main.py`: The entry point for the CLI.
*   `ingestor.py`: Core logic for traversing and reading files.
*   `utils.py`: Helper functions (e.g., text cleaning, binary detection).
*   `requirements.txt`: Minimal dependencies (e.g., `aiohttp`, `click` or `argparse`).
*   `README.md`: Instructions on how to use the tool.

### 2. **CLI Features**
The `main.py` should accept arguments for:
*   `--url`: The GitHub repository URL to ingest.
*   `--branch`: Specific branch to clone (optional, default to `main` or `master`).
*   `--output`: Output filename (default: `repository_context.md`).
*   `--max-lines`: Limit processing to avoid massive files (handle 1k, 10k, 100k scales).

### 3. **Core Functionality (The "Ingestor")**
*   **Downloading:** efficient cloning or downloading of the repo (zip download via `aiohttp` is preferred for speed if strictly pulling).
*   **Traversal:** Recursively walk the directory tree.
*   **Filtering:** Automatically exclude:
    *   `.git/` directory
    *   Image files (`.png`, `.jpg`, etc.)
    *   Binary files (executables, `.pyc`)
    *   Lock files (`package-lock.json` if > 1k lines)
*   **Markdown Formatting:**
    *   The output file must be structured for LLM readability.
    *   **Header:** Repository Metadata (Name, URL, Date).
    *   **File Tree:** A text-based tree diagram of the included files.
    *   **Content:**
        ```markdown
        ## File: path/to/file.py
        ```python
        <file_content>
        ```
        ```

### 4. **Handling Scale (1k vs 10k vs 100k lines)**
*   **1k (Small):** Ingest everything.
*   **10k (Medium):** Standard ingestion. Add a Table of Contents.
*   **100k (Large):**
    *   Implement a "safety brake" or confirmation prompt.
    *   Offer a `--structure-only` mode that only dumps the directory tree and docstrings.
    *   Consider splitting output into multiple parts if token count is projected to exceed standard context windows.

### 5. **Asynchronous Implementation**
*   Use `asyncio` to read files concurrently where possible to speed up the aggregation of thousands of small files.

# DELIVERABLE:
Generate the code for the files listed in the **Project Structure**. Ensure the code is production-ready, error-handled, and fully documented.
```
