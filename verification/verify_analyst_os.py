
import os
from playwright.sync_api import sync_playwright, expect

def verify_analyst_os():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context()
        page = context.new_page()

        # Construct the file path to the HTML file
        file_path = os.path.abspath("showcase/analyst_os.html")
        url = f"file://{file_path}"

        print(f"Navigating to: {url}")
        page.goto(url)

        # Wait for the desktop to be visible
        desktop = page.locator("#desktop")
        expect(desktop).to_be_visible()
        print("Desktop is visible.")

        # Click on the DCF icon in the dock
        dcf_icon = page.locator(".dock-item[data-title='DCF Valuator']")
        dcf_icon.click()
        print("Clicked DCF icon.")

        # Verify DCF window opens
        dcf_window = page.locator("#win-dcf")
        expect(dcf_window).to_be_visible()
        print("DCF Window is visible.")

        # Click on the LBO icon in the dock
        lbo_icon = page.locator(".dock-item[data-title='LBO Modeler']")
        lbo_icon.click()
        print("Clicked LBO icon.")

        # Verify LBO window opens
        lbo_window = page.locator("#win-lbo")
        expect(lbo_window).to_be_visible()
        print("LBO Window is visible.")

        # Wait a moment for iframes to render content (though screenshot might capture them blank if cross-origin strictness applies to file://, but usually local files can see each other in same dir)
        page.wait_for_timeout(2000)

        # Take a screenshot
        screenshot_path = "verification/analyst_os.png"
        page.screenshot(path=screenshot_path)
        print(f"Screenshot saved to {screenshot_path}")

        browser.close()

if __name__ == "__main__":
    verify_analyst_os()
