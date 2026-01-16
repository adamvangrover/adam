from playwright.sync_api import sync_playwright
import os
import time

def run():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        # Grant permissions for clipboard etc if needed, but mainly we want to allow cross-origin for CDN if possible
        context = browser.new_context()
        page = context.new_page()

        # Navigate to the archive page
        cwd = os.getcwd()
        url = f"file://{cwd}/showcase/market_mayhem_archive.html"
        print(f"Navigating to {url}")
        page.goto(url)

        # Wait for chart to render (simple delay)
        time.sleep(2)

        # Take a screenshot of the archive page
        page.screenshot(path="verification/archive_enhanced.png", full_page=True)
        print("Screenshot saved to verification/archive_enhanced.png")

        # Click the first 'ACCESS' button
        try:
            page.wait_for_selector(".read-btn", timeout=5000)
            buttons = page.query_selector_all(".read-btn")
            if buttons:
                print(f"Found {len(buttons)} reports. Clicking the first one...")
                buttons[0].click()

                # Wait for navigation
                page.wait_for_load_state('domcontentloaded')
                time.sleep(2)

                page.screenshot(path="verification/report_enhanced.png", full_page=True)
                print("Screenshot saved to verification/report_enhanced.png")
            else:
                print("No reports found!")
        except Exception as e:
            print(f"Interaction failed: {e}")

        browser.close()

if __name__ == "__main__":
    run()
