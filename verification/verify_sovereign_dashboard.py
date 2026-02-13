import time
import subprocess
import os
import sys
from playwright.sync_api import sync_playwright

def verify_dashboard():
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

            # Navigate to Dashboard
            page.goto("http://localhost:8000/showcase/sovereign_dashboard.html")

            # Verify Title
            assert page.title() == "Sovereign Credit Dashboard"
            print("Title verified.")

            # Wait for Ticker List to populate
            page.wait_for_selector(".ticker-item")

            # Click MSFT
            page.click("text=MSFT")
            time.sleep(1) # Allow fetch and render

            # Verify Quant Data Loaded (Look for Revenue)
            page.wait_for_selector("text=Revenue")
            # Verify specific mock value for MSFT Revenue (211,915 M)
            # The dashboard formats it as "$211,915 M"
            assert page.is_visible("text=211,915")
            print("MSFT Quantitative Data verified.")

            # Verify Risk Data Loaded
            page.wait_for_selector("text=EXECUTIVE SUMMARY")
            page.wait_for_selector("text=COVENANT ANALYSIS")
            print("MSFT Risk Data verified.")

            # Take Screenshot
            os.makedirs("verification_screenshots", exist_ok=True)
            screenshot_path = "verification_screenshots/sovereign_dashboard_msft.png"
            page.screenshot(path=screenshot_path)
            print(f"Screenshot saved to {screenshot_path}")

            browser.close()

    finally:
        server_process.kill()

if __name__ == "__main__":
    verify_dashboard()
