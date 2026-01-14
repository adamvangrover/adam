from playwright.sync_api import sync_playwright
import time
import os

def verify_dashboard():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        print("Navigating directly to Unified Banking Dashboard...")
        try:
            page.goto("http://localhost:8000/showcase/unified_banking.html", timeout=10000)
        except Exception as e:
            print(f"Failed to load page: {e}")
            return

        print("Navigated to Unified Banking Dashboard.")

        # Wait for network graph to render
        time.sleep(3)

        # Verify Scenario Selector exists
        selector = page.get_by_label("SCENARIO") # Or just check select element
        # Since I didn't use a label tag explicitly linked, I'll find by css
        scenario_dropdown = page.locator("#scenario-select")
        if scenario_dropdown.is_visible():
            print("Scenario selector found.")
        else:
            print("WARNING: Scenario selector not found.")

        # Take screenshot
        output_path = "verification/dashboard_final_screenshot.png"
        page.screenshot(path=output_path)
        print(f"Screenshot saved to {output_path}")

        browser.close()

if __name__ == "__main__":
    verify_dashboard()
