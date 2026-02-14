import time
import subprocess
import os
import sys
from playwright.sync_api import sync_playwright

def verify_market_mayhem_v24():
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
            page.goto("http://localhost:8000/showcase/market_mayhem_archive_v24.html")

            # 1. Verify Title
            title = page.title()
            print(f"Page Title: {title}")
            assert "MARKET MAYHEM ARCHIVE" in title
            print("Title verified.")

            # 2. Verify System Monitor
            page.wait_for_selector(".system-monitor")
            regime = page.text_content("#sys-regime")
            print(f"Market Regime: {regime}")
            assert "THE GREAT BIFURCATION" in regime
            print("System Monitor verified.")

            # 3. Verify Concept Cloud
            page.wait_for_selector("#conceptCloud .tag-cloud-item")
            tags = page.query_selector_all("#conceptCloud .tag-cloud-item")
            print(f"Concept Tags found: {len(tags)}")
            assert len(tags) > 1
            print("Concept Cloud verified.")

            # 4. Verify Conviction Meters
            # These are dynamically injected, so wait a bit
            time.sleep(1)
            meters = page.query_selector_all(".archive-item div[style*='background: rgba(0, 0, 0, 0.2)']")
            print(f"Conviction Meters found: {len(meters)}")
            assert len(meters) > 0
            print("Conviction Meters verified.")

            # 5. Verify Chart
            assert page.is_visible("#sentimentChart")
            print("Chart verified.")

            # 6. Verify New Bifurcation Links
            print("Verifying new Bifurcation links...")
            assert page.is_visible("a[href='report_great_bifurcation_2026.html']")
            assert page.is_visible("a[href='system2_critique_bifurcation.html']")
            assert page.is_visible("a[href='agent_alignment_log_feb_2026.html']")
            print("Bifurcation links verified.")

            # Take Screenshot
            os.makedirs("verification_screenshots", exist_ok=True)
            screenshot_path = "verification_screenshots/market_mayhem_v24.png"
            page.screenshot(path=screenshot_path)
            print(f"Screenshot saved to {screenshot_path}")

            browser.close()

    finally:
        server_process.kill()

if __name__ == "__main__":
    verify_market_mayhem_v24()
