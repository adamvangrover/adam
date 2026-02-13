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

            # Check for new columns
            page.wait_for_selector("text=SNC Rating", timeout=5000)
            page.wait_for_selector("text=DRC", timeout=5000)
            page.wait_for_selector("text=LTV", timeout=5000)
            page.wait_for_selector("text=Conviction", timeout=5000)

            print("Annex C Columns Verified!")

            # Check for data
            page.wait_for_selector("text=Pass", timeout=5000)
            # Check for tooltips text content (hidden but present in DOM)
            if page.locator("text=Capacity").count() > 0:
                 print("Found DRC Data")
            if page.locator("text=LTV").count() > 0:
                 print("Found LTV Data")

            # Screenshot
            os.makedirs("verification/screenshots", exist_ok=True)
            page.screenshot(path="verification/screenshots/credit_memo_snc_features.png")
            print("Screenshot saved to verification/screenshots/credit_memo_snc_features.png")

        except Exception as e:
            print(f"VERIFICATION FAILED: {e}")
            os.makedirs("verification/screenshots", exist_ok=True)
            page.screenshot(path="verification/screenshots/credit_memo_snc_failed.png")
            print("Screenshot saved to verification/screenshots/credit_memo_snc_failed.png")
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
