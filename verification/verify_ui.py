from playwright.sync_api import sync_playwright
import time

def verify_showcase():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        # 1. Navigate to Mission Control
        print("Navigating to Mission Control...")
        # Use domcontentloaded instead of networkidle which can be flaky
        page.goto("http://localhost:8000/showcase/index.html")
        page.wait_for_load_state("domcontentloaded")

        # Verify stats are visible
        print("Checking stats...")
        page.wait_for_selector("#cpu-stat")

        # Take screenshot of Dashboard
        print("Taking Dashboard screenshot...")
        page.screenshot(path="verification/dashboard.png")

        # 2. Check Live Link Toggle
        print("Checking Live Link toggle...")
        try:
            toggle = page.wait_for_selector("#live-toggle", timeout=5000)
            if toggle:
                print("Toggle found.")
        except:
            print("Toggle NOT found (or timed out).")

        # 3. Navigate to Reports
        print("Navigating to Reports...")
        page.goto("http://localhost:8000/showcase/reports.html")
        page.wait_for_load_state("domcontentloaded")

        # Verify reports loaded
        print("Verifying reports loaded...")
        try:
            page.wait_for_selector("#report-list div", timeout=5000)

            # Click first report
            print("Clicking first report...")
            page.click("#report-list div:first-child")
            time.sleep(1) # wait for render

            # Take screenshot of Reports
            print("Taking Reports screenshot...")
            page.screenshot(path="verification/reports.png")
        except Exception as e:
            print(f"Failed to verify reports: {e}")
            page.screenshot(path="verification/reports_error.png")

        browser.close()

if __name__ == "__main__":
    verify_showcase()
