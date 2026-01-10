from playwright.sync_api import sync_playwright, expect
import time

def verify_showcase():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        # Navigate to the local server
        page.goto("http://0.0.0.0:8080/showcase/index.html")

        # 1. Verify Shell Loads
        print("Verifying Shell...")
        # Use more specific locator to avoid strict mode violation
        expect(page.locator("text=ADAM").first).to_be_visible()
        expect(page.locator("#toggle-terminal")).to_be_visible()

        # 2. Verify Initial Iframe (Mission Control)
        print("Verifying Mission Control Iframe...")
        iframe = page.frame_locator("#content-frame")
        expect(iframe.locator("text=Mission Control").first).to_be_visible()

        # 3. Test Navigation to Reports
        print("Navigating to Reports...")
        page.click("a[href='#reports']")
        time.sleep(1) # Allow iframe update
        expect(iframe.locator("text=Generated Reports")).to_be_visible()

        # 4. Open Terminal
        print("Opening Terminal...")
        page.click("#toggle-terminal")
        expect(page.locator("#terminal-pane")).to_be_visible()

        # 5. Take Screenshot
        print("Taking Screenshot...")
        page.screenshot(path="verification/showcase_final.png")

        browser.close()

if __name__ == "__main__":
    verify_showcase()
