from playwright.sync_api import sync_playwright
import os
import sys

def verify_forecast_features():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        page.on("console", lambda msg: print(f"CONSOLE: {msg.text}"))

        cwd = os.getcwd()
        file_path = f"file://{cwd}/showcase/credit_memo_automation.html"

        print(f"Navigating to: {file_path}")
        page.goto(file_path)

        # Wait for initial load
        page.wait_for_timeout(3000)

        # 0. Select a Memo from Library
        print("Selecting first memo from library...")
        first_item = page.locator("#library-list > div").first
        if first_item.is_visible():
            first_item.click()
            print("Clicked first library item.")
            # Wait for generation simulation (approx 4.5s) + buffer
            page.wait_for_timeout(8000)
        else:
            print("FAIL: Library list empty.")
            sys.exit(1)

        # 1. Open Forecast Tab
        print("Clicking Forecast Tab...")
        page.locator("#btn-tab-forecast").click()
        page.wait_for_timeout(2000)

        # 3. Check Container
        container = page.locator("#forecast-container")
        if container.is_visible():
            content = container.inner_text()
            print(f"Container Content: '{content[:500]}...'")

            if "Debt Repayment Forecast" in content:
                print("PASS: 'Debt Repayment Forecast' text found.")
            else:
                print("FAIL: 'Debt Repayment Forecast' text missing.")

            if "Facility Ratings" in content:
                print("PASS: 'Facility Ratings' text found.")
            else:
                print("FAIL: 'Facility Ratings' text missing.")
        else:
            print("FAIL: Forecast Container not visible.")
            # Check if tab is visible
            if page.locator("#tab-forecast").is_visible():
                print("DEBUG: Tab is visible, but container is not.")
            else:
                print("DEBUG: Tab is NOT visible.")

        browser.close()

if __name__ == "__main__":
    verify_forecast_features()
