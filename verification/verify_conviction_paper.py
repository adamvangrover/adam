import json
import os
import sys
import subprocess
import time
from playwright.sync_api import sync_playwright

def verify_conviction_paper():
    # 1. Verify JSON Content
    print("Verifying JSON content...")
    try:
        with open("showcase/data/market_mayhem_index.json", "r") as f:
            data = json.load(f)

        titles = [item["title"] for item in data]
        expected_titles = [
            "MARKET MAYHEM: THE NIXON SHOCK",
            "MARKET MAYHEM: THE FLASH CRASH",
            "MARKET MAYHEM: THE MEME STOCK REVOLT"
        ]

        for title in expected_titles:
            if title in titles:
                print(f"Found expected title: {title}")
            else:
                print(f"ERROR: Missing expected title: {title}")
                sys.exit(1)
    except Exception as e:
        print(f"Error verifying JSON: {e}")
        sys.exit(1)

    # 2. Verify UI Elements via Playwright
    print("Verifying UI elements...")

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

            # Navigate to the page
            page.goto("http://localhost:8000/showcase/market_mayhem_conviction.html")

            # Wait for content to load
            try:
                page.wait_for_selector(".paper-sheet", timeout=5000)
                print("Content loaded successfully.")
            except Exception as e:
                print(f"Error waiting for content: {e}")
                sys.exit(1)

            # Check for new visual elements
            if page.query_selector(".watermark"):
                print("Watermark found.")
            else:
                print("ERROR: Watermark not found.")
                sys.exit(1)

            if page.query_selector(".binder-rings-container"):
                print("Binder Rings found.")
            else:
                print("ERROR: Binder Rings not found.")
                sys.exit(1)

            # Check Conviction Stamp
            if page.query_selector(".conviction-stamp"):
                print("Conviction Stamp found.")
            else:
                print("Warning: Conviction Stamp not found (might be missing for this specific item).")

            # Take screenshot
            os.makedirs("verification_screenshots", exist_ok=True)
            screenshot_path = "verification_screenshots/conviction_paper_enhanced.png"
            page.screenshot(path=screenshot_path, full_page=True)
            print(f"Screenshot saved to {screenshot_path}")

            browser.close()

    finally:
        server_process.kill()

if __name__ == "__main__":
    verify_conviction_paper()
