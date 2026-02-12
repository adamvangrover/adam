from playwright.sync_api import sync_playwright
import time
import os

def verify_modules():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        # 1. Verify Scenario Lab
        print("Verifying Scenario Lab...")
        page.goto("http://localhost:8000/showcase/scenario_lab.html")
        page.wait_for_load_state("networkidle")

        # Check title
        title = page.title()
        print(f"Scenario Lab Title: {title}")
        if "Scenario Lab" not in title:
            print("ERROR: Scenario Lab title mismatch")

        # Check if chart exists
        if page.locator("#scenarioChart").count() > 0:
            print("Chart canvas found")
        else:
            print("ERROR: Chart canvas not found")

        # Check if table has rows (meaning data loaded)
        try:
            page.wait_for_selector("#scenarioTableBody tr", timeout=5000)
            rows = page.locator("#scenarioTableBody tr").count()
            print(f"Found {rows} scenarios in table")
            if rows == 0:
                 print("ERROR: No scenarios loaded")
        except:
            print("ERROR: Timed out waiting for table rows")

        page.screenshot(path="verification_screenshots/scenario_lab.png")
        print("Saved screenshot to verification_screenshots/scenario_lab.png")

        # 2. Verify Policy Center
        print("\nVerifying Policy Center...")
        page.goto("http://localhost:8000/showcase/policy_center.html")
        page.wait_for_load_state("networkidle")

        title = page.title()
        print(f"Policy Center Title: {title}")

        # Check for global rules
        try:
            page.wait_for_selector("#global-rules .group", timeout=5000)
            if page.locator("#global-rules .group").count() > 0:
                print("Global rules rendered")
            else:
                print("ERROR: Global rules not rendered")
        except:
             print("ERROR: Timed out waiting for global rules")

        # Check for sector grid
        if page.locator("#sector-grid > div").count() > 0:
            print("Sector grid rendered")
        else:
            print("ERROR: Sector grid not rendered")

        page.screenshot(path="verification_screenshots/policy_center.png")
        print("Saved screenshot to verification_screenshots/policy_center.png")

        # 3. Verify Index Links
        print("\nVerifying Index Links...")
        page.goto("http://localhost:8000/showcase/index.html")

        # Click Scenario Lab link (first one found)
        # We need to make sure we click the right one if duplicates exist (but we removed them)
        # Use text locator
        with page.expect_navigation():
            page.get_by_text("OPEN LAB", exact=False).click()
        print("Navigated to Scenario Lab from Index")

        page.go_back()

        # Click Policy Center link
        with page.expect_navigation():
            page.get_by_text("VIEW POLICIES", exact=False).click()
        print("Navigated to Policy Center from Index")

        browser.close()

if __name__ == "__main__":
    os.makedirs("verification_screenshots", exist_ok=True)
    verify_modules()
