import sys
import os
import time
from http.server import HTTPServer, SimpleHTTPRequestHandler
import threading
from playwright.sync_api import sync_playwright

# Constants
PORT = 8085
URL = f"http://localhost:{PORT}/showcase/credit_memo_automation.html"

def serve():
    # Serve directly from repo root so /showcase paths work
    print(f"Serving from {os.getcwd()} at port {PORT}")
    server_address = ('', PORT)
    httpd = HTTPServer(server_address, SimpleHTTPRequestHandler)
    httpd.serve_forever()

def verify():
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()

        try:
            print(f"Verifying: {URL}")
            page.goto(URL)

            # 1. Wait for Library
            print("Waiting for Library...")
            page.wait_for_selector("#library-list > div", timeout=10000)

            # 2. Click Apple Inc. (Assuming it's in the list, usually 2nd or 3rd)
            print("Selecting Apple Inc...")
            # Find element with text "Apple Inc."
            apple_btn = page.locator("text=Apple Inc.").first
            apple_btn.click()

            # 3. Check for Credit Ratings in Header
            print("Checking Credit Ratings...")
            # Look for "Moody's" or "S&P" text
            page.wait_for_selector("text=Moody's", timeout=5000)
            print("Found Credit Ratings!")

            # 4. Click Annex C Tab
            print("Clicking Annex C Tab...")
            page.click("#btn-tab-annex-c")

            # 5. Check Content
            print("Checking Annex C Content...")
            # Ensure tab is visible
            page.wait_for_selector("#tab-annex-c", state="visible", timeout=2000)

            # Check for headers
            if page.locator("text=Capital Structure").count() > 0:
                 print("Found Capital Structure Header")
            else:
                 print("Capital Structure Header NOT FOUND")

            if page.locator("text=Equity Market Data").count() > 0:
                 print("Found Equity Market Data Header")
            else:
                 print("Equity Market Data Header NOT FOUND")

            page.wait_for_selector("text=Capital Structure", timeout=5000)
            page.wait_for_selector("text=Equity Market Data", timeout=5000)
            page.wait_for_selector("text=Debt Facilities", timeout=5000)

            print("Annex C Content Verified!")

            # Screenshot
            os.makedirs("verification/screenshots", exist_ok=True)
            page.screenshot(path="verification/screenshots/credit_memo_new_features.png")
            print("Screenshot saved to verification/screenshots/credit_memo_new_features.png")

        except Exception as e:
            print(f"VERIFICATION FAILED: {e}")
            os.makedirs("verification/screenshots", exist_ok=True)
            page.screenshot(path="verification/screenshots/credit_memo_failed.png")
            print("Screenshot saved to verification/screenshots/credit_memo_failed.png")
            sys.exit(1)
        finally:
            browser.close()

if __name__ == "__main__":
    # Start Server in Thread
    t = threading.Thread(target=serve, daemon=True)
    t.start()

    # Give server a moment
    time.sleep(2)

    verify()
    print("VERIFICATION SUCCESS")
    sys.exit(0)
