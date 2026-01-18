import time
import subprocess
import threading
from playwright.sync_api import sync_playwright

def run_server():
    subprocess.run(["python3", "-m", "http.server", "8000"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def verify_dashboard():
    # Start server in background
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()

    # Give server time to start
    time.sleep(2)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        try:
            print("Navigating to dashboard...")
            page.goto("http://localhost:8000/showcase/unified_banking_dashboard.html")

            # Wait for data to load and chart to render
            # The chart has id 'gammaChart'
            print("Waiting for chart...")
            page.wait_for_selector("#gammaChart")

            # Additional wait to ensure Chart.js animation finishes or at least starts
            time.sleep(2)

            print("Taking screenshot...")
            page.screenshot(path="verification_gamma.png")
            print("Screenshot saved to verification_gamma.png")

        except Exception as e:
            print(f"Error: {e}")
        finally:
            browser.close()

if __name__ == "__main__":
    verify_dashboard()
