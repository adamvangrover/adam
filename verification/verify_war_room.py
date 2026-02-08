
from playwright.sync_api import sync_playwright
import os

def run():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        # Load the HTML file directly
        file_path = os.path.abspath("showcase/war_room_v2.html")
        page.goto(f"file://{file_path}")

        # Wait for the ticker to render
        page.wait_for_selector("#ticker-feed .ticker-row")

        # Take a screenshot
        screenshot_path = "verification/war_room_v2.png"
        page.screenshot(path=screenshot_path)
        print(f"Screenshot saved to {screenshot_path}")

        browser.close()

if __name__ == "__main__":
    run()
