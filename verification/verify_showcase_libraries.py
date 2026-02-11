import os
import sys
import time
import subprocess
import threading
from playwright.sync_api import sync_playwright

# Define the directory to serve (repository root, so showcase/ works)
DOC_ROOT = os.getcwd()
PORT = 8085

def start_server():
    # Start Python's built-in HTTP server
    subprocess.run([sys.executable, "-m", "http.server", str(PORT)], cwd=DOC_ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def verify_libraries():
    # Start server in background
    server_thread = threading.Thread(target=start_server, daemon=True)
    server_thread.start()

    # Give server a moment to start
    time.sleep(2)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        # 1. Daily Briefings
        print("Verifying Daily Briefings Library...")
        page.goto(f"http://localhost:{PORT}/showcase/daily_briefings_library.html")
        page.wait_for_timeout(2000) # Wait for fetch
        page.screenshot(path="verification/screenshot_daily_briefings.png")
        print("  - Screenshot saved: verification/screenshot_daily_briefings.png")

        # 2. Market Pulse
        print("Verifying Market Pulse Library...")
        page.goto(f"http://localhost:{PORT}/showcase/market_pulse_library.html")
        page.wait_for_timeout(2000)
        page.screenshot(path="verification/screenshot_market_pulse.png")
        print("  - Screenshot saved: verification/screenshot_market_pulse.png")

        # 3. House View
        print("Verifying House View Library...")
        page.goto(f"http://localhost:{PORT}/showcase/house_view_library.html")
        page.wait_for_timeout(2000)
        page.screenshot(path="verification/screenshot_house_view.png")
        print("  - Screenshot saved: verification/screenshot_house_view.png")

        # 4. Portfolio Dashboard
        print("Verifying Portfolio Dashboard...")
        page.goto(f"http://localhost:{PORT}/showcase/portfolio_dashboard.html")
        page.wait_for_timeout(2000)
        page.screenshot(path="verification/screenshot_portfolio.png")
        print("  - Screenshot saved: verification/screenshot_portfolio.png")

        # 5. Archive Hub Update
        print("Verifying Archive Hub...")
        page.goto(f"http://localhost:{PORT}/showcase/market_mayhem_archive.html")
        page.wait_for_timeout(1000)
        page.screenshot(path="verification/screenshot_archive_hub.png")
        print("  - Screenshot saved: verification/screenshot_archive_hub.png")

        browser.close()

if __name__ == "__main__":
    verify_libraries()
