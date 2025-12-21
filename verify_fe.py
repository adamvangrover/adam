
from playwright.sync_api import expect, sync_playwright


def verify_financial_engineering():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        # Navigate to the page
        # Using localhost:8000 assuming the server is running from root
        url = "http://localhost:8000/showcase/financial_engineering.html"
        print(f"Navigating to {url}")
        page.goto(url)

        # Wait for the page to load
        # Updated to match v24.0 "ADAM ENGINE"
        expect(page.locator("h1")).to_contain_text("ADAM")
        expect(page.locator("h1")).to_contain_text("ENGINE")

        # Take a screenshot of the initial state (DCF Tab)
        print("Taking initial screenshot...")
        page.screenshot(path="verification_dcf.png")

        # Click on Monte Carlo Risk tab
        print("Clicking Monte Carlo Tab...")
        # v24.0 uses id="btn-mc"
        page.click("button#btn-mc")

        # Wait for the tab to be visible
        expect(page.locator("#tab-mc")).to_be_visible()

        # Run Simulation
        print("Running Monte Carlo Simulation...")
        # v24.0 uses button with text "RUN MONTE CARLO"
        page.click("button:text('RUN MONTE CARLO')")

        # Wait for simulation to finish (loading hidden)
        expect(page.locator("#mc-loading")).to_be_hidden(timeout=10000)

        # Verify results appeared
        expect(page.locator("#mc-mean")).not_to_have_text("-")

        # Take screenshot of Monte Carlo results
        print("Taking Monte Carlo screenshot...")
        page.screenshot(path="verification_monte_carlo.png")

        # Click on Sensitivity Matrix tab
        print("Clicking Sensitivity Tab...")
        # v24.0 uses id="btn-sensitivity"
        page.click("button#btn-sensitivity")
        expect(page.locator("#tab-sensitivity")).to_be_visible()

        # Take screenshot of Sensitivity Matrix
        print("Taking Sensitivity screenshot...")
        page.screenshot(path="verification_sensitivity.png")

        browser.close()

if __name__ == "__main__":
    verify_financial_engineering()
