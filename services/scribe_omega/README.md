# Scribe-Omega

**Scribe-Omega** is a portable, high-performance microservice designed to convert code repositories into optimized Markdown and text formats for LLM ingestion.

## Architecture

It is built as a "Single-Binary, Multi-Interface" tool:
- **CLI**: High-performance terminal tool.
- **Web App**: Lightweight HTMX/Tailwind dashboard.

## Tiered Scaling Logic ("The Lens")

| Tier | Logic Strategy |
|---|---|
| **Micro (1k)** | Structural Only. Extracts README, configs, and skeleton code (signatures only). |
| **Standard (10k)** | Logic Dense. Includes code, strips comments/whitespace. Excludes tests/vendor. |
| **Full (100k)** | Full Fidelity. Recursive export of everything except binaries/.git. |

## Usage

### CLI

```bash
# Standard scan of current directory
go run main.go

# Specific tier and output file
go run main.go --tier micro --output summary.md

# Specific directory
go run main.go --dir /path/to/repo --tier full
```

### Web Server

```bash
go run main.go serve --port 8080
```
Open `http://localhost:8080` in your browser.

## Prompting

This tool is designed to be used by other Agents.
**Prompt for Agents**: "Use Scribe-Omega to digest the repository at [PATH] using the [TIER] lens."
