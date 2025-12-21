
import os

from playwright.sync_api import sync_playwright


def run():
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        # Navigate to the local file
        file_path = os.path.abspath("showcase/newsletter_market_mayhem.html")
        page.goto(f"file://{file_path}")

        # Take a screenshot
        page.screenshot(path="verification/newsletter.png", full_page=True)
        browser.close()

if __name__ == "__main__":
    run()
