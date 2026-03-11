import os
from playwright.sync_api import sync_playwright

def run():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context()
        page = context.new_page()

        file_path = os.path.abspath("showcase/market_mayhem_archive.html")
        print(f"Navigating to: file://{file_path}")
        page.goto(f"file://{file_path}")

        page.wait_for_timeout(2000)

        # Check for new strategic command elements
        try:
            page.wait_for_selector("#strategicPanel :text('ACTIONABLE PLANS')", timeout=5000)
            print("Actionable plans rendered in Strategic Panel.")
        except Exception as e:
            print(f"Actionable plans verification failed: {e}")

        # Save screenshot
        screenshot_path = "tests/market_mayhem/archive_v2_overlays_ui.png"
        page.screenshot(path=screenshot_path)
        print(f"Saved screenshot to {screenshot_path}")

        browser.close()

if __name__ == "__main__":
    run()