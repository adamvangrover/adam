import os
import time
import subprocess
import sys
from playwright.sync_api import sync_playwright

def verify_crisis_sim():
    # Start HTTP Server
    server_process = subprocess.Popen([sys.executable, "-m", "http.server", "8000"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("Started HTTP server on port 8000")
    time.sleep(2)

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(viewport={'width': 1200, 'height': 2000})
            page = context.new_page()

            # Navigate to the 2008 report which should have a simulation
            url = "http://localhost:8000/showcase/newsletter_market_mayhem_sep_2008.html"
            print(f"Navigating to {url}")
            page.goto(url)

            # Check for Crisis Simulation Header
            print("Checking for Crisis Simulation Header...")
            header = page.get_by_role("heading", name="5. Crisis Simulation:")
            if header.is_visible():
                print("PASS: Crisis Simulation Header visible.")
            else:
                print("FAIL: Crisis Simulation Header missing.")

            # Check for Crisis Chart Canvas
            print("Checking for Crisis Chart Canvas...")
            canvas = page.locator("#crisisSimChart")
            if canvas.is_visible():
                print("PASS: Crisis Chart Canvas visible.")
            else:
                print("FAIL: Crisis Chart Canvas missing.")

            # Take Screenshot
            screenshot_path = "verification/crisis_sim_verified.png"
            page.screenshot(path=screenshot_path, full_page=True)
            print(f"Screenshot saved to {screenshot_path}")

            browser.close()

    except Exception as e:
        print(f"Error: {e}")
    finally:
        server_process.terminate()
        print("Server stopped.")

if __name__ == "__main__":
    verify_crisis_sim()
