import os
from playwright.sync_api import sync_playwright

def run():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context()
        page = context.new_page()

        file_path = os.path.abspath("showcase/dashboard.html")
        print(f"Navigating to: file://{file_path}")
        page.goto(f"file://{file_path}")

        # Wait for the page to fully load and run JS
        page.wait_for_timeout(2000)

        # Check if the title renders
        try:
            page.wait_for_selector("h1:has-text('ADAM v23.5')", timeout=5000)
            print("Title rendered.")
        except Exception as e:
            print(f"Title verification failed: {e}")

        # Check if agents view renders
        try:
            page.click("button:has-text('AGENTS')")
            page.wait_for_selector("table", timeout=5000)
            print("Agents table rendered.")
        except Exception as e:
            print(f"Agents verification failed: {e}")

        # Save screenshot
        screenshot_path = "tests/market_mayhem/dashboard_v2_ui.png"
        page.screenshot(path=screenshot_path)
        print(f"Saved screenshot to {screenshot_path}")

        browser.close()

if __name__ == "__main__":
    run()