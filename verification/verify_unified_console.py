from playwright.sync_api import sync_playwright, expect
import os

def test_unified_console():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        # Load the local HTML file
        page.goto("file:///app/showcase/unified_credit_console.html")

        # Verify title
        expect(page).to_have_title("Unified Credit & Quantum Recovery Console")

        # Verify LBO Tab is active by default
        expect(page.locator("#tab-lbo")).to_be_visible()
        expect(page.locator("#metric-leverage")).to_contain_text("5.00x")

        # Switch to ABL Tab
        page.click("text=ABL & Working Capital")
        expect(page.locator("#tab-abl")).to_be_visible()
        expect(page.locator("#avail-total")).not_to_be_empty()

        # Switch to Quantum Tab
        page.click("text=Quantum Recovery Search")
        expect(page.locator("#tab-quantum")).to_be_visible()

        # Take screenshot
        page.screenshot(path="verification/unified_console.png")

        print("Verification complete. Screenshot saved.")
        browser.close()

if __name__ == "__main__":
    test_unified_console()
