from playwright.sync_api import sync_playwright, expect
import os

def test_distressed_page():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        # Load the local HTML file
        page.goto("file:///app/showcase/distressed_lbo.html")

        # Verify title (Updated to AVG)
        expect(page).to_have_title("AVG Distressed Credit Pricing Console")

        # Verify initial calculation (Wait for JS to run)
        # 50M EBITDA * 6.0 EV = 300M EV
        # Debt: 200 (Sen) + 100 (Jun) + 50 (Mez) = 350M
        # Leverage: 350/50 = 7.0x

        expect(page.locator("#metric-leverage")).to_contain_text("7.00x")

        # Take screenshot
        page.screenshot(path="verification/distressed_lbo_dashboard_avg.png")

        print("Verification complete. Screenshot saved.")
        browser.close()

if __name__ == "__main__":
    test_distressed_page()
