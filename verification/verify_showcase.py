
from playwright.sync_api import sync_playwright
import os

def run():
    # Ensure mock data is present
    if not os.path.exists("showcase/js/mock_data.js"):
        print("Mock data missing!")
        return

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        # Load the index page
        # Using file:// protocol requires absolute path
        abs_path = os.path.abspath("showcase/index.html")
        page.goto(f"file://{abs_path}")

        # Wait for data load
        page.wait_for_timeout(2000)

        # Take screenshot of Mission Control
        page.screenshot(path="verification/mission_control.png")
        print("Mission Control screenshot taken.")

        # Navigate to Reports
        page.click("text=v21.0 Synchronous") # Clicking the card that leads to reports.html
        page.wait_for_timeout(2000)
        page.screenshot(path="verification/reports.png")
        print("Reports screenshot taken.")

        browser.close()

if __name__ == "__main__":
    run()
