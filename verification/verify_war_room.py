import time
import subprocess
import os
from playwright.sync_api import sync_playwright

def verify_war_room():
    # Start HTTP server
    # Use port 8000
    proc = subprocess.Popen(["python3", "-m", "http.server", "8000"], cwd="showcase")
    time.sleep(3) # Wait for server

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.set_viewport_size({"width": 1920, "height": 1080})

            # Navigate
            url = "http://localhost:8000/war_room_v2.html"
            print(f"Navigating to {url}")
            page.goto(url)

            # Wait for dynamic content
            print("Waiting for dynamic content...")
            page.wait_for_timeout(5000)

            # Check for key elements
            if page.locator("#ticker-feed").is_visible():
                print("[PASS] Ticker Feed Visible")
            else:
                print("[FAIL] Ticker Feed Not Visible")

            if page.locator("#riskCanvas").is_visible():
                print("[PASS] Risk Radar Canvas Visible")
            else:
                print("[FAIL] Risk Radar Canvas Not Visible")

            # Screenshot
            os.makedirs("verification_screenshots", exist_ok=True)
            path = "verification_screenshots/war_room_v2.png"
            page.screenshot(path=path)
            print(f"Screenshot saved to {path}")
    except Exception as e:
        print(f"Verification failed: {e}")
    finally:
        proc.terminate()

if __name__ == "__main__":
    verify_war_room()
