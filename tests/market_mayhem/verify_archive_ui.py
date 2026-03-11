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

        # Wait for the page to fully load and run JS
        page.wait_for_timeout(2000)

        # Check if static data was loaded successfully (the controller should now find it and not fail on fetch)
        try:
            # Check if strategic panel has content other than "Loading Strategic Command..."
            page.wait_for_selector("#strategicPanel:not(:has-text('Loading Strategic Command...'))", timeout=5000)
            print("Strategic panel rendered dynamically.")
        except Exception as e:
            print(f"Strategic panel verification failed: {e}")

        try:
            # Check if archive grid has items
            page.wait_for_selector("#archiveGrid .archive-item", timeout=5000)
            print("Archive grid rendered items.")
        except Exception as e:
            print(f"Archive grid verification failed: {e}")

        # Save screenshot
        screenshot_path = "tests/market_mayhem/archive_v2_ui.png"
        page.screenshot(path=screenshot_path)
        print(f"Saved screenshot to {screenshot_path}")

        browser.close()

if __name__ == "__main__":
    run()