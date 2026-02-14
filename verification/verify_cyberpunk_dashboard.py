import time
import subprocess
import os
import sys
from playwright.sync_api import sync_playwright

def verify_cyberpunk_dashboard():
    # Start HTTP Server
    server_process = subprocess.Popen(
        [sys.executable, "-m", "http.server", "8001"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        cwd=os.getcwd()
    )
    time.sleep(2) # Wait for server to start

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()

            # Navigate to the target static newsletter
            target_url = "http://localhost:8001/showcase/newsletter_market_mayhem_jan_31_2026.html"
            print(f"Navigating to {target_url}...")
            page.goto(target_url)

            # Wait for JS to execute (DOM injections)
            time.sleep(1)

            # 1. Verify Layout Injection
            print("Verifying Dashboard Container...")
            container_exists = page.evaluate("document.querySelector('.dashboard-container') !== null")
            assert container_exists, "Dashboard Container not found!"
            print("Dashboard Container verified.")

            # 2. Verify System 2 Terminal
            print("Verifying System 2 Terminal...")
            terminal_exists = page.is_visible(".system2-terminal")
            assert terminal_exists, "System 2 Terminal not visible!"
            print("System 2 Terminal verified.")

            # 3. Verify Metadata Header
            print("Verifying Metadata Header...")
            header_exists = page.is_visible(".metadata-header")
            assert header_exists, "Metadata Header not visible!"
            print("Metadata Header verified.")

            # 4. Verify Source Chips
            print("Verifying Source Chips...")
            chips_count = page.evaluate("document.querySelectorAll('.source-chip').length")
            print(f"Source Chips found: {chips_count}")
            assert chips_count > 0, "No Source Chips found!"
            print("Source Chips verified.")

            # 5. Verify Conviction Gauge
            print("Verifying Conviction Gauge...")
            gauge_exists = page.is_visible(".conviction-bar")
            assert gauge_exists, "Conviction Gauge not visible!"
            print("Conviction Gauge verified.")

            # Take Screenshot
            os.makedirs("verification_screenshots", exist_ok=True)
            screenshot_path = "verification_screenshots/cyberpunk_dashboard_verify.png"
            page.screenshot(path=screenshot_path)
            print(f"Screenshot saved to {screenshot_path}")

            browser.close()

    except Exception as e:
        print(f"Verification Failed: {e}")
        sys.exit(1)

    finally:
        server_process.kill()

if __name__ == "__main__":
    verify_cyberpunk_dashboard()
