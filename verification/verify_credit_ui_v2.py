import os
import sys
import time
import subprocess
import requests
from playwright.sync_api import sync_playwright

def verify_credit_ui_v2():
    print("Starting Credit UI Verification (v2 Cyberpunk)...")

    # 1. Start Server
    port = 8090
    server_process = subprocess.Popen(
        [sys.executable, "-m", "http.server", str(port)],
        cwd=os.path.abspath("showcase"),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    # Wait for server
    time.sleep(2)

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()

            url = f"http://localhost:{port}/credit_memo.html"
            print(f"Navigating to {url}...")
            page.goto(url)

            # 2. Check Initial State
            assert "ADAM" in page.title()
            print("PASS: Page title verified.")

            # 3. Check Selector Population (Wait for option to be attached)
            print("Waiting for data load...")
            page.wait_for_selector("option[value='Apple Inc.']", state="attached", timeout=5000)
            print("PASS: Selector populated (Apple Inc. found).")

            # 4. Select Apple and Generate
            print("Selecting Apple Inc. and generating...")
            page.select_option("#borrower-select", value="Apple Inc.")
            page.click("#generate-btn")

            # 5. Wait for Agent Trace & Memo
            print("Monitoring Agent Terminal...")
            # Wait for "Analysis Complete" in logs.
            try:
                page.wait_for_function("document.body.innerText.includes('Analysis Complete')", timeout=20000)
            except Exception as e:
                print("Timeout waiting for analysis completion.")
                print("Terminal Content:", page.inner_text("#agent-terminal"))
                raise e
            print("PASS: Agent workflow completed.")

            # 6. Check Memo Content
            # Wait for memo content to be visible
            page.wait_for_selector("#memo-content", state="visible", timeout=2000)
            content = page.inner_text("#memo-content")

            # Case insensitive check or check for Uppercase due to CSS
            assert "APPLE INC." in content
            assert "TICKER: AAPL" in content
            assert "EBITDA" in content
            print("PASS: Memo content verified.")

            # 7. Click Citation
            print("Clicking citation...")
            # Find the first citation tag and click it
            citation = page.locator(".citation-tag").first
            citation.click()

            # 8. Check Evidence Viewer
            print("Checking Evidence Viewer...")
            # Check for BBox Highlight in the canvas
            page.wait_for_selector(".bbox-highlight", state="visible", timeout=3000)

            # Check if label is correct
            label_el = page.locator(".bbox-label").first
            label = label_el.inner_text()
            print(f"PASS: Evidence highlighted. Label: {label}")

            # Take Screenshot
            os.makedirs("verification_screenshots", exist_ok=True)
            page.screenshot(path="verification_screenshots/credit_ui_v2.png")
            print("Screenshot saved to verification_screenshots/credit_ui_v2.png")

    except Exception as e:
        print(f"FAIL: UI Verification failed: {e}")
        # Take debug screenshot if page is alive
        if 'page' in locals():
            try:
                os.makedirs("verification_screenshots", exist_ok=True)
                page.screenshot(path="verification_screenshots/credit_ui_v2_fail.png")
            except:
                pass
        raise e
    finally:
        server_process.kill()
        print("Server stopped.")

if __name__ == "__main__":
    verify_credit_ui_v2()
