from playwright.sync_api import sync_playwright
import time

def run():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        # Navigate to the page
        page.goto("http://localhost:8000/showcase/market_mayhem_conviction.html")

        # Wait for the loading to finish and content to appear
        # We look for .paper-sheet which is added dynamically
        try:
            page.wait_for_selector(".paper-sheet", timeout=5000)
            print("Content loaded successfully.")
        except Exception as e:
            print(f"Error waiting for content: {e}")
            # Take screenshot anyway to see what happened (e.g. loading error)

        # Take screenshot
        screenshot_path = "verification_screenshots/conviction_paper.png"
        page.screenshot(path=screenshot_path, full_page=True)
        print(f"Screenshot saved to {screenshot_path}")

        browser.close()

if __name__ == "__main__":
    run()
