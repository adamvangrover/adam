import os
from playwright.sync_api import sync_playwright

def run():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        # Load the local HTML file
        path = os.path.abspath("showcase/market_mayhem_archive.html")
        page.goto(f"file://{path}")

        # Wait for content to load
        page.wait_for_selector(".archive-item")

        # Take a full page screenshot
        page.screenshot(path="verification/market_mayhem_archive.png", full_page=True)
        print("Screenshot taken: verification/market_mayhem_archive.png")

        browser.close()

if __name__ == "__main__":
    run()
