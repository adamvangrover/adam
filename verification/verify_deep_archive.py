
from playwright.sync_api import sync_playwright
import os

def run():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        # Navigate to the archive page
        cwd = os.getcwd()
        url = f"file://{cwd}/showcase/market_mayhem_archive.html"
        print(f"Navigating to {url}")
        page.goto(url)

        # Take a screenshot of the archive page
        page.screenshot(path="verification/archive_deep_dive.png", full_page=True)
        print("Screenshot of deep archive saved to verification/archive_deep_dive.png")

        # Click on a new file (e.g., September 2025)
        # We look for the text "THE LIQUIDITY TRAP"
        page.click("text=THE LIQUIDITY TRAP")

        # Take a screenshot of the newsletter page
        page.screenshot(path="verification/newsletter_sep_2025.png", full_page=True)
        print("Screenshot of Sept newsletter saved to verification/newsletter_sep_2025.png")

        browser.close()

if __name__ == "__main__":
    run()
