# Agent Development Patterns & Practices

> **"Consistency is the bedrock of scalability."**

This document outlines the standard patterns, workflows, and architectural decisions used in the Adam v26.0 repository. It complements `AGENTS.md` by focusing on the "How-To" of daily development.

---

## 1. The Verification Workflow (Playwright)

We do not trust; we verify. Every frontend change requires visual proof.

### The Loop
1.  **Change:** Modify HTML/JS/CSS.
2.  **Server:** Start a local HTTP server (`python3 -m http.server 8000`).
3.  **Script:** Write a Playwright script in `verification/`.
    *   *Must* use headless mode.
    *   *Must* take a screenshot.
    *   *Must* verify critical elements (e.g., canvas rendering, button clicks).
4.  **Verify:** Inspect the screenshot.
5.  **Commit:** Only once visually confirmed.

### Standard Script Template (`verification/verify_template.py`)

```python
import os
import time
import subprocess
from playwright.sync_api import sync_playwright

def verify_feature():
    # 1. Start Server
    server = subprocess.Popen(["python3", "-m", "http.server", "8000"],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(2)

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()

            # 2. Navigate
            page.goto("http://localhost:8000/showcase/your_page.html")

            # 3. Assert & Interact
            page.wait_for_selector("#critical-element")

            # 4. Capture
            page.screenshot(path="verification_feature.png")

            browser.close()
    finally:
        server.terminate()

if __name__ == "__main__":
    verify_feature()
```

---

## 2. Repo Metadata & The "Repo Graph"

We treat our code as data. The repository structure is introspected to provide live documentation.

*   **Generator:** `scripts/generate_repo_metadata.py` scans the codebase.
*   **Artifact:** `showcase/data/repo_metadata.json` contains the graph.
*   **Viewer:** `showcase/agent_dev_hub.html` visualizes it.

### Tagging Convention
To ensure your agent is correctly indexed:
1.  **Class Name:** Must end in `Agent` or `Analyst`.
2.  **Docstring:** Must be descriptive.
3.  **File Location:** Must be within `core/agents/`.

---

## 3. Data-Driven Navigation

Hardcoded links are technical debt. We use a centralized `site_map.json` to drive navigation.

*   **Source:** `showcase/site_map.json`
*   **Consumer:** `showcase/js/nav.js` (AdamNavigator)
*   **Visualizer:** `showcase/network_map.html`

**Adding a Page:**
1.  Create the HTML file.
2.  Add an entry to `showcase/site_map.json` under the appropriate category.
3.  The Navigation Bar and Network Map update automatically.

---

## 4. The "System 2" Frontend Pattern

High-tier reports use a specific interactivity pattern to simulate "Thinking".

### Components
*   **Virtual Toolbar:** Floats at the bottom/side. Contains "Verify", "Redact", "Print".
*   **System 2 Overlay:** A full-screen overlay that plays a decryption/analysis animation before revealing the content.
*   **Raw Source Viewer:** A modal that displays the underlying JSON data with syntax highlighting.

### Implementation
Refer to `showcase/js/market_mayhem_viewer.js` for the reference implementation.

---

## 5. Mock Data Strategy

To ensure the showcase works without a live backend:
1.  **Artifacts:** Store JSON snapshots in `showcase/data/`.
2.  **Loaders:** Use `fetch` with a fallback to `window.MOCK_DATA` if defined.
3.  **Hybrid Mode:** The `AdamNavigator` detects if the API is available and switches modes.

```javascript
async function loadData() {
    try {
        const res = await fetch('/api/data');
        if (!res.ok) throw new Error("API Offline");
        return await res.json();
    } catch (e) {
        console.warn("Using Mock Data");
        return window.MOCK_DATA; // Fallback
    }
}
```
