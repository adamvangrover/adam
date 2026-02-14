import time
import subprocess
import os
import sys
from playwright.sync_api import sync_playwright

def verify_system2_monitor():
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

            # Navigate to High Conviction Monitor
            page.goto("http://localhost:8000/showcase/high_conviction_monitor.html")

            # 1. Verify Title
            title = page.title()
            print(f"Page Title: {title}")
            assert "HIGH CONVICTION MONITOR" in title
            print("Title verified.")

            # 2. Verify Card Injection
            page.wait_for_selector(".asset-card")
            cards = page.query_selector_all(".asset-card")
            print(f"Asset Cards found: {len(cards)}")
            assert len(cards) > 0
            print("Asset cards verified.")

            # 3. Verify Judge Section
            judge_section = page.query_selector(".judge-section")
            assert judge_section is not None
            print("System 2 Judge section verified.")

            # 4. Verify Score Calculation (Dynamic)
            # Check if a score badge exists and has content
            score_badge = page.query_selector(".conviction-badge")
            score_text = score_badge.inner_text()
            print(f"First Conviction Score: {score_text}")
            assert "/100" in score_text
            print("Score calculation verified.")

            # Take Screenshot
            os.makedirs("verification_screenshots", exist_ok=True)
            screenshot_path = "verification_screenshots/system2_monitor.png"
            page.screenshot(path=screenshot_path)
            print(f"Screenshot saved to {screenshot_path}")

            browser.close()

    finally:
        server_process.kill()

if __name__ == "__main__":
    verify_system2_monitor()
