from playwright.sync_api import sync_playwright, expect
import os

def test_quantum_page():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        # Load the local HTML file
        # Assuming the repo root is /app, so absolute path is /app/showcase/quantum_search.html
        page.goto("file:///app/showcase/quantum_search.html")

        # Verify title
        expect(page).to_have_title("AdamVanGrover Quantum Search Console")

        # Verify Chart.js canvas exists
        expect(page.locator("#scheduleChart")).to_be_visible()
        expect(page.locator("#lossChart")).to_be_visible()

        # Take screenshot of initial state
        page.screenshot(path="verification/quantum_console_initial.png")

        print("Verification complete. Screenshot saved.")
        browser.close()

if __name__ == "__main__":
    test_quantum_page()
