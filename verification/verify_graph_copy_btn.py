import time
import subprocess
import os
from playwright.sync_api import sync_playwright

def run():
    server = subprocess.Popen(["python3", "-m", "http.server", "8001"], cwd="showcase")
    time.sleep(2)

    os.makedirs("verification_artifacts", exist_ok=True)

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()

            # Go to page
            page.goto("http://localhost:8001/system_brain.html")
            print("Page loaded")
            time.sleep(5) # Wait for graph to stabilize

            # Search for "HOUSE VIEW" to select it
            print("Searching for HOUSE VIEW...")
            page.fill("#search-input", "HOUSE VIEW")
            time.sleep(2)

            # Check for button
            btn = page.query_selector("button:has-text('Copy Context')")
            if btn:
                print("Copy Button Found!")
                # Click it
                btn.click()
                time.sleep(1)
            else:
                print("Copy Button NOT Found.")
                # Capture what IS there
                print(page.inner_html("#inspector"))

            page.screenshot(path="verification_artifacts/verify_copy_btn.png")
            print("Screenshot saved to verification_artifacts/verify_copy_btn.png")

    finally:
        server.terminate()

if __name__ == "__main__":
    run()
