import os
import sys
import time
import subprocess
from playwright.sync_api import sync_playwright

def verify_credit_features():
    print("Starting Comprehensive Credit Features Verification...")

    # 1. Start Server
    port = 8092
    server_process = subprocess.Popen(
        [sys.executable, "-m", "http.server", str(port)],
        cwd=os.path.abspath("showcase"),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    # Wait for server
    time.sleep(2)

    page = None
    browser = None
    playwright = None

    try:
        playwright = sync_playwright().start()
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page()

        # --- Test 1: Credit Memo Automation (Enterprise) ---
        print("\n[Test 1] Credit Memo Automation (Enterprise)")
        url = f"http://localhost:{port}/credit_memo_automation.html"
        page.goto(url)

        # Check for Global Controls
        print("Checking for global controls...")
        page.wait_for_selector("button:has-text('Edit Mode')", state="attached", timeout=5000)
        page.wait_for_selector("label:has-text('Load JSON')", state="attached")
        page.wait_for_selector("button:has-text('Save JSON')", state="attached")
        print("PASS: Global controls found.")

        # Test Edit Mode Toggle
        print("Testing Edit Mode...")
        page.click("button:has-text('Edit Mode')")

        # Wait for "Done Editing"
        page.wait_for_selector("button:has-text('Done Editing')", state="attached")
        print("PASS: Edit mode toggled on.")

        page.click("button:has-text('Done Editing')")

        # Wait for "Edit Mode" again
        page.wait_for_selector("button:has-text('Edit Mode')", state="attached")
        print("PASS: Edit mode toggled off.")

        # --- Test 2: Credit Memo V2 (Cyberpunk) ---
        print("\n[Test 2] Credit Memo V2 (Cyberpunk)")
        url = f"http://localhost:{port}/credit_memo.html"
        page.goto(url)

        # Check for Edit/Upload/Save buttons
        page.wait_for_selector("#edit-btn", state="attached")
        page.wait_for_selector("#export-btn", state="attached")
        page.wait_for_selector("#upload-json", state="attached")
        print("PASS: Cyberpunk controls found.")

        # --- Test 3: Sovereign Dashboard (Upload) ---
        print("\n[Test 3] Sovereign Dashboard")
        url = f"http://localhost:{port}/sovereign_dashboard.html"
        page.goto(url)

        # Check for File Input
        page.wait_for_selector("#fileInput", state="attached")
        print("PASS: File upload input found.")

        # Take Screenshot
        os.makedirs("verification_screenshots", exist_ok=True)
        page.screenshot(path="verification_screenshots/comprehensive_features_pass.png")
        print("Screenshot saved.")

    except Exception as e:
        print(f"FAIL: Verification failed: {e}")
        if page:
            try:
                os.makedirs("verification_screenshots", exist_ok=True)
                page.screenshot(path="verification_screenshots/feature_fail.png")
            except:
                pass
        raise e
    finally:
        if browser:
            browser.close()
        if playwright:
            playwright.stop()
        server_process.kill()
        print("Server stopped.")

if __name__ == "__main__":
    verify_credit_features()
