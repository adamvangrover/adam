import os
import subprocess
import time
from playwright.sync_api import sync_playwright

def verify_frontend():
    # Start server
    server = subprocess.Popen(["python3", "-m", "http.server", "8000", "--directory", "showcase"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(2) # Wait for start

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()

            # Mock API
            page.route("**/api/simulation/start", lambda route: route.fulfill(
                status=200,
                body='{"status": "started", "state": {"turn": 1, "lcr": 115.0, "cet1": 13.5, "sovereign_spread": 450, "repo_haircut": 2.0, "counterparty_cds": 150, "intraday_liquidity": 5000, "history": ["Simulation Started"], "injects": []}}',
                content_type="application/json"
            ))

            page.route("**/api/simulation/state", lambda route: route.fulfill(
                status=200,
                body='{"turn": 1, "lcr": 115.0, "cet1": 13.5, "sovereign_spread": 450, "repo_haircut": 2.0, "counterparty_cds": 150, "intraday_liquidity": 5000, "history": ["Simulation Started"], "injects": []}',
                content_type="application/json"
            ))

            # Load page from server
            page.goto("http://localhost:8000/crisis_simulator.html")

            # Click Start
            page.click("text=INITIALIZE SIMULATION")

            # Wait for overlay to hide
            page.wait_for_selector("#start-overlay", state="hidden")

            # Wait for data update
            page.wait_for_function("document.getElementById('val-lcr').innerText.includes('115')")

            # Take Screenshot
            page.screenshot(path="verification/crisis_sim_updated.png")
            print("Screenshot saved to verification/crisis_sim_updated.png")

            browser.close()
    finally:
        server.kill()

if __name__ == "__main__":
    verify_frontend()
