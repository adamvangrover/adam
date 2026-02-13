import os
import sys
import time
import subprocess
import requests
from playwright.sync_api import sync_playwright

def verify_credit_live_integration():
    print("Starting Credit Live Integration Verification...")

    # 1. Start Server (HTTP for frontend)
    port = 8092
    server_process = subprocess.Popen(
        [sys.executable, "-m", "http.server", str(port)],
        cwd=os.path.abspath("showcase"),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    time.sleep(2)

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()

            url = f"http://localhost:{port}/credit_memo.html"
            print(f"Navigating to {url}...")
            page.goto(url)

            # 2. Check Initial Load (Library should load)
            print("Waiting for library load...")
            # 'Goldman Sachs' is in the new library.
            # We use state="attached" because options in a select might not be "visible" if dropdown is closed.
            page.wait_for_selector("option[value='Goldman Sachs']", state="attached", timeout=5000)
            print("PASS: Full Library loaded (Goldman Sachs found).")

            # 3. Trigger Generation (Fallback Path)
            print("Selecting Goldman Sachs and generating...")
            page.select_option("#borrower-select", value="Goldman Sachs")

            # Intercept the API call to confirm it happens
            api_called = False
            def handle_request(route):
                nonlocal api_called
                if "/api/credit/generate" in route.request.url:
                    print("API Call Intercepted!")
                    api_called = True
                    # Return error to trigger fallback
                    route.fulfill(status=500, body='{"status":"error"}')
                else:
                    route.continue_()

            page.route("**/api/credit/generate", handle_request)

            page.click("#generate-btn")

            # Wait a bit for JS to process the failure
            page.wait_for_timeout(2000)

            if api_called:
                print("PASS: UI attempted to call /api/credit/generate.")
            else:
                print("FAIL: UI did not attempt API call.")

            # 4. Check Mode Indicator
            # Should be CACHED_ARTIFACT_MODE since we aborted the request
            # Wait for it to change
            try:
                page.wait_for_function("document.querySelector('.header-nav span span').innerText.includes('CACHED')", timeout=5000)
            except:
                pass

            indicator = page.inner_text(".header-nav span span")
            print(f"Mode Indicator: {indicator}")
            assert "CACHED" in indicator
            print("PASS: Fallback mode indicated.")

            # 5. Check Content Render
            # Increase timeout since simulation runs in frontend
            # The log in JS says simulation takes 800+1000+1200 ms = 3s + overhead
            page.wait_for_selector("#memo-content", state="visible", timeout=20000)
            content = page.inner_text("#memo-content")
            # The mocked memo content should be there
            # Goldman Sachs
            if "Goldman Sachs" in content:
                print("PASS: Content rendered from cache.")
            else:
                print(f"FAIL: Content mismatch. Found: {content[:100]}...")

            # Take Screenshot
            os.makedirs("verification_screenshots", exist_ok=True)
            page.screenshot(path="verification_screenshots/credit_live_fallback.png")

    except Exception as e:
        print(f"FAIL: Verification failed: {e}")
        # Only try screenshot if we can
        try:
            if 'page' in locals():
                page.screenshot(path="verification_screenshots/credit_live_fail.png")
        except:
            pass
        raise e
    finally:
        server_process.kill()
        print("Server stopped.")

if __name__ == "__main__":
    verify_credit_live_integration()
