import os
import sys
from playwright.sync_api import sync_playwright

def verify_showcase():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        # Construct file URL
        repo_root = os.path.abspath(os.getcwd())
        file_url = f"file://{repo_root}/showcase/index.html"

        print(f"Navigating to {file_url}")
        page.goto(file_url)

        # Check for the new elements
        human_lane = page.locator("h3", has_text="HUMAN DASHBOARD")
        machine_lane = page.locator("h3", has_text="MACHINE INTERFACE")

        if human_lane.is_visible() and machine_lane.is_visible():
            print("SUCCESS: Human and Machine Dashboard options found.")
        else:
            print("FAILURE: Dashboard options not found.")
            sys.exit(1)

        # Take screenshot of the main area
        screenshot_path = "verification_showcase.png"
        page.screenshot(path=screenshot_path)
        print(f"Screenshot saved to {screenshot_path}")

        browser.close()

if __name__ == "__main__":
    verify_showcase()
