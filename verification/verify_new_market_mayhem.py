import os
import sys
import time
import subprocess
import threading
from playwright.sync_api import sync_playwright

# Define the directory to serve (repository root, so showcase/ works)
DOC_ROOT = os.getcwd()
PORT = 8086

def start_server():
    # Start Python's built-in HTTP server
    subprocess.run([sys.executable, "-m", "http.server", str(PORT)], cwd=DOC_ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def verify_new_files():
    # Start server in background
    server_thread = threading.Thread(target=start_server, daemon=True)
    server_thread.start()

    # Give server a moment to start
    time.sleep(2)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        # 1. Check MM09192025.html
        print("Verifying showcase/MM09192025.html...")
        page.goto(f"http://localhost:{PORT}/showcase/MM09192025.html")
        page.wait_for_timeout(1000)
        title = page.title()
        assert "Adam's Weekly Dispatch: The Great Disconnect" in title, f"Unexpected title: {title}"
        page.screenshot(path="verification/screenshot_MM09192025.png")
        print("  - Passed. Screenshot saved.")

        # 2. Check MM06292025.html
        print("Verifying showcase/MM06292025.html...")
        page.goto(f"http://localhost:{PORT}/showcase/MM06292025.html")
        page.wait_for_timeout(1000)
        title = page.title()
        assert "Market Mayhem - Interactive Weekly Briefing" in title, f"Unexpected title: {title}"
        page.screenshot(path="verification/screenshot_MM06292025.png")
        print("  - Passed. Screenshot saved.")

        # 3. Check SNC_Guide.html
        print("Verifying showcase/SNC_Guide.html...")
        page.goto(f"http://localhost:{PORT}/showcase/SNC_Guide.html")
        page.wait_for_timeout(1000)
        title = page.title()
        assert "SNC Exam Preparation Guide" in title, f"Unexpected title: {title}"
        page.screenshot(path="verification/screenshot_SNC_Guide.png")
        print("  - Passed. Screenshot saved.")

        # 4. Check market_mayhem_archive.html for new entries
        print("Verifying showcase/market_mayhem_archive.html new entries...")
        page.goto(f"http://localhost:{PORT}/showcase/market_mayhem_archive.html")
        page.wait_for_timeout(1000)
        content = page.content()
        assert "Adam's Weekly Dispatch: The Great Disconnect" in content, "MM09192025 entry missing in archive"
        assert "Market Mayhem - Interactive Weekly Briefing" in content, "MM06292025 entry missing in archive"
        page.screenshot(path="verification/screenshot_market_mayhem_archive_updated.png")
        print("  - Passed. Screenshot saved.")

        browser.close()

if __name__ == "__main__":
    verify_new_files()
