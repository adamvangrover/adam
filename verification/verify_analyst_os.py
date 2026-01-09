from playwright.sync_api import sync_playwright
import os

def run():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        # Determine the absolute path to the showcase/analyst_os.html file
        # Assuming the script is running from the repo root
        file_path = os.path.abspath("showcase/analyst_os.html")
        url = f"file://{file_path}"

        print(f"Navigating to {url}")
        page.goto(url)

        # Wait for the Dock to appear
        page.wait_for_selector(".dock")

        # Take a screenshot of the entire desktop
        page.screenshot(path="verification/analyst_os.png")
        print("Screenshot saved to verification/analyst_os.png")

        # Take a screenshot specifically of the Dock to verify new icons
        dock = page.locator(".dock")
        dock.screenshot(path="verification/dock_icons.png")
        print("Dock screenshot saved to verification/dock_icons.png")

        browser.close()

if __name__ == "__main__":
    run()
