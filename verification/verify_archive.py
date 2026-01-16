from playwright.sync_api import sync_playwright, expect
import time

def verify_archive():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        # 1. Archive Page
        print("Navigating to Archive...")
        page.goto("http://localhost:8000/showcase/market_mayhem_archive.html")

        # Verify title
        expect(page).to_have_title("ADAM v23.5 :: MARKET MAYHEM ARCHIVE")

        # Verify presence of new items
        print("Checking for new items...")
        deep_dive_link = page.locator("text=Deep Dive Report: December 10, 2025").first
        expect(deep_dive_link).to_be_visible()

        page.screenshot(path="verification/archive_new.png")
        print("Archive screenshot saved.")

        # 2. Navigate to Report
        print("Clicking report link...")
        # The link is the "DECRYPT ->" button, but the title text is in the h3.
        # The structure is: div.archive-item > a.read-btn
        # We can find the item by text, then find the link inside it.

        # Or easier: click the DECRYPT button inside the item with that title
        # locator filtering
        item = page.locator(".archive-item").filter(has_text="Deep Dive Report: December 10, 2025")
        item.locator("a.read-btn").click()

        # 3. Verify Report Page
        print("Verifying Report Page...")
        expect(page).to_have_title("ADAM v23.5 :: 🤿 Deep Dive Report: December 10, 2025")
        expect(page.locator("h1.title")).to_contain_text("Deep Dive Report")

        # Verify content from the MD
        expect(page.locator("text=Financial Overview")).to_be_visible()

        page.screenshot(path="verification/report_new.png")
        print("Report screenshot saved.")

        browser.close()

if __name__ == "__main__":
    verify_archive()
