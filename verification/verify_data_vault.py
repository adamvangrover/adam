from playwright.sync_api import sync_playwright
import time
import os

def run():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        # 1. Load the Data Vault
        print("Loading page...")
        page.goto("http://localhost:8000/showcase/data.html")
        page.wait_for_load_state("networkidle")

        # 2. Wait for tree to populate (check for file count)
        print("Waiting for data...")
        page.wait_for_selector("#file-count", state="visible")
        time.sleep(2) # Allow JS to process the large JSON

        # 3. Take initial screenshot (Code View default)
        print("Capturing Code View...")
        os.makedirs("verification", exist_ok=True)
        page.screenshot(path="verification/data_vault_code_view.png")

        # 4. Interact: Click a file (e.g., README.md)
        print("Clicking a file...")
        # Use a more specific selector if possible, or just click the first file item that isn't a folder
        # We need to find a file. Let's look for README.md if visible, or just the first item.
        try:
            # Expand root if needed (it's flat usually at top level)
            page.click("div[data-path='README.md']")
            time.sleep(1)
            page.screenshot(path="verification/data_vault_file_selected.png")
        except:
            print("README.md not found directly, clicking first available file.")
            files = page.query_selector_all(".file-tree-item:not(.folder)")
            if files:
                files[0].click()
                time.sleep(1)
                page.screenshot(path="verification/data_vault_file_selected.png")

        # 5. Switch to Graph View
        print("Switching to Graph View...")
        page.click("#btn-mode-graph")
        time.sleep(3) # Wait for Vis.js stabilization
        page.screenshot(path="verification/data_vault_graph_view.png")

        # 6. Switch to Notebook View
        print("Switching to Notebook View...")
        # Select a python file first to make it meaningful
        try:
            # Try to find a .py file
            py_file = page.query_selector("div[data-path$='.py']")
            if py_file:
                py_file.click()
                time.sleep(0.5)
        except:
            pass

        page.click("#btn-mode-notebook")
        time.sleep(1)
        page.screenshot(path="verification/data_vault_notebook_view.png")

        # 7. Test Neural Link
        print("Testing Neural Link...")
        page.fill("#chat-input", "Explain this code")
        page.click("button:has-text('Neural Link')") # Focus tab if needed, but input is visible
        # Click send button icon
        page.click("#assistant-panel button .fa-paper-plane")
        time.sleep(2) # Wait for simulated response
        page.screenshot(path="verification/data_vault_chat.png")

        browser.close()
        print("Verification complete.")

if __name__ == "__main__":
    run()
