import time
import subprocess
import os
import sys
from playwright.sync_api import sync_playwright

def verify_market_mayhem():
    # Start HTTP Server
    server_process = subprocess.Popen(
        [sys.executable, "-m", "http.server", "8000"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        cwd=os.getcwd()
    )
    time.sleep(2) # Wait for server to start

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()

            # Navigate to Archive
            page.goto("http://localhost:8000/showcase/market_mayhem_archive.html")

            # 1. Verify Title
            title = page.title()
            print(f"Page Title: {title}")
            assert "MARKET MAYHEM ARCHIVE" in title
            print("Title verified.")

            # 2. Verify Chart
            assert page.is_visible("#sentimentChart")
            print("Chart verified.")

            # 3. Check for specific content or DOM changes
            print("Verifying sidebar panels...")
            assert page.query_selector(".sidebar-title:has-text('STRATEGIC COMMAND')")
            assert page.query_selector(".sidebar-title:has-text('FORWARD OUTLOOK')")
            assert page.query_selector(".sidebar-title:has-text('TOP CONVICTION')")
            assert page.query_selector(".sidebar-title:has-text('WATCH LIST')")
            assert page.query_selector(".sidebar-title:has-text('SECTOR RISK')")
            print("Sidebar panels verified.")

            print("Verifying grid...")
            assert page.query_selector("#archiveGrid")
            print("Grid verified.")

            # Take Screenshot
            os.makedirs("verification_screenshots", exist_ok=True)
            screenshot_path = "verification_screenshots/market_mayhem_archive.png"
            page.screenshot(path=screenshot_path, full_page=True)
            print(f"Screenshot saved to {screenshot_path}")

            browser.close()

    finally:
        server_process.kill()

if __name__ == "__main__":
    verify_market_mayhem()
