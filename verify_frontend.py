from playwright.sync_api import sync_playwright, expect
import os
import time

def verify_archive():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        # Navigate to the archive page
        page.goto("http://localhost:8080/showcase/market_mayhem_archive.html")

        # Check Title
        expect(page).to_have_title("ADAM v23.5 :: MARKET MAYHEM ARCHIVE")

        # Check for 2026 Header
        header = page.locator('.year-header[data-year="2026"]')
        expect(header).to_be_visible()

        # Check for the report
        report_title = page.locator('.item-title').filter(has_text="GLOBAL MACRO-STRATEGIC OUTLOOK 2026")
        expect(report_title).to_be_visible()

        # Check Filter
        option = page.locator('#yearFilter option[value="2026"]')
        expect(option).to_have_count(1)

        # Take Screenshot
        os.makedirs("verification_images", exist_ok=True)
        page.screenshot(path="verification_images/archive_2026.png", full_page=True)
        print("Verification successful. Screenshot saved.")

        browser.close()

if __name__ == "__main__":
    verify_archive()
