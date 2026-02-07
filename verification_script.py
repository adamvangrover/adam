import os
from playwright.sync_api import sync_playwright

def run():
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()

        # 1. Verify Archive Page
        cwd = os.getcwd()
        archive_url = f"file://{cwd}/showcase/market_mayhem_archive.html"
        print(f"Navigating to {archive_url}")
        page.goto(archive_url)

        # Take a screenshot of the top (Should show 2026 entry)
        page.screenshot(path="archive_top.png")
        print("Screenshot of archive top taken.")

        # Scroll to bottom to find 1929 entry
        page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        page.screenshot(path="archive_bottom.png")
        print("Screenshot of archive bottom taken.")

        # 2. Verify 2026 Newsletter
        newsletter_2026_url = f"file://{cwd}/showcase/newsletter_market_mayhem_mar_2026.html"
        print(f"Navigating to {newsletter_2026_url}")
        page.goto(newsletter_2026_url)
        page.screenshot(path="newsletter_2026.png")
        print("Screenshot of 2026 newsletter taken.")

        # 3. Verify 1929 Newsletter
        newsletter_1929_url = f"file://{cwd}/showcase/newsletter_market_mayhem_oct_1929.html"
        print(f"Navigating to {newsletter_1929_url}")
        page.goto(newsletter_1929_url)
        page.screenshot(path="newsletter_1929.png")
        print("Screenshot of 1929 newsletter taken.")

        browser.close()

if __name__ == "__main__":
    run()
