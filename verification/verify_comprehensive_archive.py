import os
import sys
import time
import subprocess
import threading
from playwright.sync_api import sync_playwright

# Define the directory to serve (repository root, so showcase/ works)
DOC_ROOT = os.getcwd()
PORT = 8087

def start_server():
    # Start Python's built-in HTTP server
    subprocess.run([sys.executable, "-m", "http.server", str(PORT)], cwd=DOC_ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def verify_comprehensive_archive():
    # Start server in background
    server_thread = threading.Thread(target=start_server, daemon=True)
    server_thread.start()

    # Give server a moment to start
    time.sleep(2)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        print("Verifying showcase/market_mayhem_archive.html content...")
        page.goto(f"http://localhost:{PORT}/showcase/market_mayhem_archive.html")
        page.wait_for_timeout(1000)

        # 1. Verify Special Collections Links
        content = page.content()
        assert "DATA VAULT" in content, "Data Vault link missing from sidebar"
        assert "GLITCH MONITOR" in content, "Glitch Monitor link missing from sidebar"

        # 2. Verify Key Report Entries
        # MM09192025
        assert "Adam's Weekly Dispatch: The Great Disconnect" in content, "MM09192025 entry missing"
        # MM06292025
        assert "Market Mayhem - Interactive Weekly Briefing" in content, "MM06292025 entry missing"
        # SNC Guide
        assert "SNC Exam Preparation Guide" in content, "SNC Guide entry missing"

        # 3. Verify Filter Functionality (Basic)
        # Select '2026' and check if items are visible (this is hard to assert without complex logic, just taking a screenshot)
        page.select_option("#yearFilter", "2026")
        page.wait_for_timeout(500)
        page.screenshot(path="verification/screenshot_archive_2026_filter.png")

        # Screenshot Full Page
        page.screenshot(path="verification/screenshot_archive_comprehensive.png", full_page=True)
        print("  - Passed. Screenshots saved.")

        browser.close()

if __name__ == "__main__":
    verify_comprehensive_archive()
