import os
from playwright.sync_api import sync_playwright

def verify_quantum_expansion():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        # 1. Verify Dashboard with New Panel
        cwd = os.getcwd()
        dashboard_url = f"file://{cwd}/showcase/quantum_search.html"
        print(f"Loading {dashboard_url}...")

        page.goto(dashboard_url)

        # Wait for simulation to "converge" (the loop in JS takes 1000 steps, roughly 3-4 seconds)
        # We can wait for the 'market-state' text to change from "CALCULATING..."
        print("Waiting for simulation convergence...")
        try:
            # Wait for text NOT to be CALCULATING... with a timeout of 10s
            # Note: Playwright doesn't have a direct 'wait_for_text_change' but we can wait for a selector state
            page.wait_for_timeout(15000)
        except Exception as e:
            print("Timeout waiting for simulation, proceeding anyway.")

        # Verify Recommender Panel Elements
        if page.locator("#market-state").is_visible():
            state_text = page.locator("#market-state").inner_text()
            print(f"Market State Visible: {state_text}")

        if page.locator("#strategy-action").is_visible():
            action_text = page.locator("#strategy-action").inner_text()
            print(f"Strategy Action Visible: {action_text}")

        # Screenshot
        print("Taking expansion dashboard screenshot...")
        page.screenshot(path="verification/quantum_expansion_dashboard.png")

        browser.close()

if __name__ == "__main__":
    if not os.path.exists("verification"):
        os.makedirs("verification")
    verify_quantum_expansion()
