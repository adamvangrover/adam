import os
from playwright.sync_api import sync_playwright

def verify_archive_visual():
    print("Capturing Market Mayhem Deep Archive Screenshot...")

    archive_path = os.path.abspath("showcase/market_mayhem_archive.html")

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(f"file://{archive_path}")

        # Wait for chart
        page.wait_for_selector("#sentimentChart", timeout=5000)

        # Take full page screenshot
        page.screenshot(path="verification/archive_enhanced.png", full_page=True)
        print("Saved verification/archive_enhanced.png")

        browser.close()

if __name__ == "__main__":
    verify_archive_visual()
