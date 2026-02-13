import os
import sys
import time
import subprocess
import requests
from playwright.sync_api import sync_playwright

def verify_credit_ui():
    print("Starting Credit UI Verification...")

    # 1. Start Server
    port = 8089
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
            # Check title
            assert "Credit Memo" in page.title()
            print("PASS: Page title verified.")

            # Check logo using locator
            logo = page.locator(".logo")
            assert logo.count() > 0
            print("PASS: Logo found.")

            # 3. Click Generate
            print("Clicking Generate Memo...")
            page.click("#generate-btn")

            # 4. Wait for Completion
            print("Waiting for simulation...")
            # Wait for memo content to appear
            try:
                page.wait_for_selector("#memo-content", state="visible", timeout=15000)
            except Exception as e:
                print("Timeout waiting for memo content.")
                # Print console logs if possible
                raise e
            print("PASS: Memo generated.")

            # 5. Check Content
            content = page.inner_text("#memo-content")
            assert "Credit Memo" in content
            assert "APPROVE" in content
            print("PASS: Memo content verified.")

            # 6. Click Citation
            print("Clicking citation...")
            # Find the first citation tag
            citation = page.locator(".citation-tag").first
            citation.click()

            # 7. Check Evidence Panel
            print("Checking Evidence Panel...")
            # Wait for panel to expand
            page.wait_for_selector("#evidence-panel.active", timeout=5000)

            # Check for BBox Highlight
            bbox = page.locator(".bbox-highlight")
            assert bbox.count() > 0
            print("PASS: BBox Highlight visible.")

            # Take Screenshot
            os.makedirs("verification_screenshots", exist_ok=True)
            page.screenshot(path="verification_screenshots/credit_ui.png")
            print("Screenshot saved to verification_screenshots/credit_ui.png")

    except Exception as e:
        print(f"FAIL: UI Verification failed: {e}")
        # Take screenshot on failure
        if 'page' in locals():
            os.makedirs("verification_screenshots", exist_ok=True)
            page.screenshot(path="verification_screenshots/credit_ui_fail.png")
        raise e
    finally:
        server_process.kill()
        print("Server stopped.")

if __name__ == "__main__":
    verify_credit_ui()
