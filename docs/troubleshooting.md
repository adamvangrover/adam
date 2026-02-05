# Troubleshooting Guide

This guide covers common issues you might encounter while installing or running Adam v26.0.

## Installation Issues

### `uv` Command Not Found
**Symptom:** `bash: uv: command not found`
**Cause:** `uv` is not in your system PATH.
**Solution:**
1.  Ensure you installed it via the official script: `curl -LsSf https://astral.sh/uv/install.sh | sh`
2.  Restart your terminal.
3.  Add it manually to your PATH if needed: `export PATH="$HOME/.cargo/bin:$PATH"`

### Missing Dependencies (e.g., `docling`, `libmagic`)
**Symptom:** `ImportError: libmagic is not available`
**Cause:** Some Python packages rely on system-level libraries.
**Solution:**
*   **Ubuntu/Debian:** `sudo apt-get install libmagic1`
*   **macOS:** `brew install libmagic`
*   **Windows:** Ensure you have the C++ Build Tools installed.

### Virtual Environment Issues
**Symptom:** `ModuleNotFoundError` even after running `uv sync`.
**Cause:** You might not have activated the virtual environment.
**Solution:**
*   Run `source .venv/bin/activate` (Linux/Mac) or `.venv\Scripts\activate` (Windows).
*   Verify your python path: `which python` should point to `.venv/bin/python`.

## Runtime Issues

### OpenAI API Errors
**Symptom:** `AuthenticationError: No API key provided`
**Cause:** The `.env` file is missing or the key is not set.
**Solution:**
1.  Check if `.env` exists in the root directory.
2.  Open `.env` and verify `OPENAI_API_KEY=sk-...` is correct (no quotes required).

### Docker Port Conflicts
**Symptom:** `Error starting userland proxy: listen tcp4 0.0.0.0:5000: bind: address already in use`
**Cause:** Another service is running on port 5000 (often AirPlay on macOS).
**Solution:**
1.  Find the process: `lsof -i :5000`
2.  Kill it: `kill <PID>`
3.  Or, change the port in `docker-compose.yml` to `5001:5000`.

### Agent "Hallucinations" or Empty Responses
**Symptom:** The agent returns "I don't know" or makes up facts.
**Cause:**
*   **Low Conviction:** The agent couldn't find reliable data.
*   **Search Failure:** The search tool (Serper/Google) failed or returned no results.
**Solution:**
*   Run with `--debug` to see the raw tool outputs.
*   Check if your search API key (if used) is valid.

## Frontend Issues

### "Network Error" in Dashboard
**Symptom:** The dashboard loads but shows "Network Error" when querying.
**Cause:** The backend Flask server is not running or CORS is blocking requests.
**Solution:**
1.  Ensure `python app.py` is running in a separate terminal.
2.  Check the browser console (F12) for specific CORS errors.

### Charts Not Rendering
**Symptom:** Blank spaces where charts should be.
**Cause:** Missing data in `data/` directory or JavaScript errors.
**Solution:**
1.  Run `python scripts/generate_mock_data.py` to populate the `data/` folder with sample files.
