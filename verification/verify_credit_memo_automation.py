import os
import sys
import time
import subprocess
from playwright.sync_api import sync_playwright

def verify_credit_memo_automation():
    print("Starting Credit Memo Automation Verification...")

    # 1. Start Server
    port = 8091
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

        url = f"http://localhost:{port}/credit_memo_automation.html"
        print(f"Navigating to {url}...")
        page.goto(url)

        # 2. Check Initial State
        title = page.title()
        assert "Enterprise Credit Memo Automation" in title, f"Unexpected title: {title}"
        print("PASS: Page title verified.")

        # 3. Check Library Population
        print("Waiting for library load...")
        # Wait for at least one library item
        page.wait_for_selector("#library-list > div", state="attached", timeout=5000)
        items = page.locator("#library-list > div")
        count = items.count()
        print(f"PASS: Library populated with {count} items.")
        assert count > 0

        # 4. Select First Item (Apple Inc usually)
        print("Selecting first library item...")
        items.first.click()

        # 5. Check Memo Content
        print("Checking Memo Content...")
        page.wait_for_selector("#memo-container h1", state="visible", timeout=2000)
        memo_title = page.inner_text("#memo-container h1")
        print(f"Memo Title: {memo_title}")
        assert len(memo_title) > 0

        # 6. Check Financials Tab
        print("Switching to Financials Tab...")
        page.click("#btn-tab-annex-a")
        # specific check for visibility of table content
        page.wait_for_selector("#financials-table tbody tr", state="visible", timeout=2000)
        rows = page.locator("#financials-table tbody tr").count()
        print(f"Financial Rows: {rows}")
        assert rows > 0

        # 7. Check DCF Tab
        print("Switching to DCF Tab...")
        page.click("#btn-tab-annex-b")
        page.wait_for_selector("#dcf-container input[type=range]", state="visible", timeout=2000)
        print("PASS: DCF Sliders visible.")

        # 8. Check Evidence Viewer Interaction
        print("Switching back to Memo Tab to check citations...")
        page.click("#btn-tab-memo")

        # Find a citation button
        citation_btn = page.locator("#memo-container button").first
        if citation_btn.count() > 0:
            print("Clicking citation button...")
            citation_btn.click()

            # Check Evidence Viewer Visibility
            # The #mock-pdf-page starts with 'hidden' class, it should be removed
            page.wait_for_selector("#mock-pdf-page:not(.hidden)", state="visible", timeout=2000)
            print("PASS: Evidence Viewer activated.")

            # Check Highlight Box
            bbox = page.locator("#highlight-box")
            assert bbox.is_visible()
            print("PASS: Highlight box visible.")
        else:
            print("WARNING: No citation buttons found to test.")

        # Take Screenshot
        os.makedirs("verification_screenshots", exist_ok=True)
        page.screenshot(path="verification_screenshots/credit_memo_automation_pass.png")
        print("Screenshot saved.")

    except Exception as e:
        print(f"FAIL: Verification failed: {e}")
        if page:
            try:
                os.makedirs("verification_screenshots", exist_ok=True)
                page.screenshot(path="verification_screenshots/credit_memo_automation_fail.png")
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
    verify_credit_memo_automation()
