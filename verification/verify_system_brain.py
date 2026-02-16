from playwright.sync_api import sync_playwright
import os
import sys
import subprocess
import time
import signal

def run():
    # Start server
    server_process = subprocess.Popen(
        [sys.executable, "-m", "http.server", "8081"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    time.sleep(2) # Give it time to start

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.set_viewport_size({"width": 1280, "height": 800})

            url = "http://localhost:8081/showcase/system_brain.html"
            print(f"Navigating to {url}")
            page.goto(url)

            # Wait for data to load and render
            try:
                page.wait_for_selector("#agent-count", timeout=5000)
                page.wait_for_selector("#synapse-list tr", timeout=5000)
                print("Data loaded.")

                # Assert knowledge percent is not default
                k_percent = page.text_content("#knowledge-percent")
                if k_percent == "--%":
                    print("FAIL: Knowledge coverage percentage is still default (--%)")
                else:
                    print(f"PASS: Knowledge coverage percentage is {k_percent}")

            except Exception as e:
                print(f"Timeout or error: {e}")

            screenshot_path = "verification/system_brain_screenshot.png"
            page.screenshot(path=screenshot_path, full_page=True)
            print(f"Screenshot saved to {screenshot_path}")

            browser.close()
    finally:
        server_process.terminate()
        server_process.wait()

if __name__ == "__main__":
    os.makedirs("verification", exist_ok=True)
    run()
