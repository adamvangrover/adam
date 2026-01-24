from playwright.sync_api import sync_playwright
import os

def test_dashboard():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        # Load the local HTML file
        file_path = os.path.abspath("showcase/unified_banking_dashboard.html")
        page.goto(f"file://{file_path}")

        # Wait for data to load (Chart.js canvas should render)
        page.wait_for_selector("#priceChart")
        page.wait_for_selector("#orderLog div") # Wait for at least one log entry

        # Take screenshot
        page.screenshot(path="verification/dashboard_screenshot.png", full_page=True)
        browser.close()

if __name__ == "__main__":
    test_dashboard()
