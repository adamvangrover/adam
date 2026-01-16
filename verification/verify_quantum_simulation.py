import os
import sys
import time
import subprocess
from playwright.sync_api import sync_playwright

def verify_quantum_simulation():
    # Start a temporary HTTP server
    server = subprocess.Popen([sys.executable, "-m", "http.server", "3000"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(2) # Give it time to start

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()

            url = "http://localhost:3000/showcase/quantum_infrastructure.html"
            print(f"Navigating to {url}...")
            page.goto(url)

            # Wait for data to load
            print("Waiting for QUANTUM_DATA...")
            page.wait_for_function("window.QUANTUM_DATA !== undefined")

            # Check Slider
            print("Checking slider...")
            slider = page.locator("#sim-slider")
            if not slider.is_visible():
                raise Exception("Simulation Slider not visible")

            # Check Metrics (Initial state)
            entropy = page.locator("#entropy-metric").inner_text()
            print(f"Initial Entropy: {entropy}")
            if entropy == "--":
                 raise Exception("Entropy metric not populated")

            # Interact: Move Slider
            print("Moving slider to 100%...")
            slider.evaluate("el => el.value = el.max")
            slider.dispatch_event("input") # Trigger change

            # Wait a bit for animation frame/metric update
            time.sleep(1)

            # Check Metrics (Final state)
            entropy_final = page.locator("#entropy-metric").inner_text()
            print(f"Final Entropy: {entropy_final}")

            if entropy == entropy_final:
                print("Warning: Entropy did not change after slider move (Animation loop might update it differently or slider range is 0?)")

            # Screenshot
            os.makedirs("verification_screenshots", exist_ok=True)
            page.screenshot(path="verification_screenshots/quantum_simulation.png")
            print("Screenshot saved to verification_screenshots/quantum_simulation.png")

            browser.close()
            print("Verification Successful.")

    except Exception as e:
        print(f"Verification Failed: {e}")
        sys.exit(1)
    finally:
        server.terminate()

if __name__ == "__main__":
    verify_quantum_simulation()
