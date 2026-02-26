from playwright.sync_api import sync_playwright
import os

def run():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        filepath = os.path.abspath("showcase/market_mayhem_archive.html")
        page.goto(f"file://{filepath}")

        page.wait_for_load_state("networkidle")

        # Take screenshot of the archive view
        page.screenshot(path="verification/archive_view.png", full_page=True)
        print("Screenshot saved to verification/archive_view.png")

        # Also check one of the new conviction reports
        filepath_conv = os.path.abspath("showcase/conviction_btc_feb26.html")
        page.goto(f"file://{filepath_conv}")
        page.screenshot(path="verification/conviction_btc.png", full_page=True)
        print("Screenshot saved to verification/conviction_btc.png")

        browser.close()

if __name__ == "__main__":
    run()
