import os
import sys
import time
import subprocess
from playwright.sync_api import sync_playwright

def verify_universal_loader():
    print("Starting Universal Loader Integration Verification...")

    # 1. Start Server
    port = 8093
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

            # --- Test 1: Credit Memo Automation (Enterprise) ---
            print("\n[Test 1] Credit Memo Automation (Enterprise)")
            url = f"http://localhost:{port}/credit_memo_automation.html"

            # Hook console to check for UniversalLoader logs
            msgs = []
            page.on("console", lambda msg: msgs.append(msg.text))

            page.goto(url)

            # Check if UniversalLoader is defined
            is_defined = page.evaluate("typeof window.UniversalLoader !== 'undefined'")
            assert is_defined, "UniversalLoader not defined on Enterprise page"
            print("PASS: UniversalLoader defined.")

            # Check if data loaded via Loader
            page.wait_for_selector("#library-list > div", state="attached", timeout=5000)
            print("PASS: Library loaded successfully.")

            # --- Test 2: Credit Memo V2 (Cyberpunk) ---
            print("\n[Test 2] Credit Memo V2 (Cyberpunk)")
            url = f"http://localhost:{port}/credit_memo.html"
            page.goto(url)

            # Check for specific success log from credit_memo_v2.js
            # "Loaded X entity profiles via UniversalLoader."
            # We wait a bit for async init
            time.sleep(1)

            # Check dropdown population
            count = page.locator("#borrower-select option").count()
            assert count > 1, "Dropdown not populated"
            print(f"PASS: Dropdown populated with {count} items.")

            # --- Test 3: Sovereign Dashboard ---
            print("\n[Test 3] Sovereign Dashboard")
            url = f"http://localhost:{port}/sovereign_dashboard.html"
            page.goto(url)

            # Check loading of Apple Inc (default)
            page.wait_for_selector("#fiscalYear", state="visible", timeout=5000)
            fy_text = page.inner_text("#fiscalYear")
            print(f"PASS: Dashboard loaded with {fy_text}")

            # Take Screenshot
            os.makedirs("verification_screenshots", exist_ok=True)
            page.screenshot(path="verification_screenshots/universal_loader_pass.png")
            print("Screenshot saved.")

    except Exception as e:
        print(f"FAIL: Verification failed: {e}")
        raise e
    finally:
        server_process.kill()
        print("Server stopped.")

if __name__ == "__main__":
    verify_universal_loader()
