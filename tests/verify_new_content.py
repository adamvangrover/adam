import os
from playwright.sync_api import sync_playwright

def verify_new_content():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        # Load the local HTML file
        cwd = os.getcwd()
        file_url = f"file://{cwd}/showcase/weekly_recap_20251210.html"
        print(f"Loading: {file_url}")

        page.goto(file_url)

        # Check title
        title = page.title()
        print(f"Title: {title}")
        assert "Market Mayhem Newsletter" in title

        # Take screenshot
        output_dir = "/home/jules/verification"
        os.makedirs(output_dir, exist_ok=True)
        screenshot_path = os.path.join(output_dir, "weekly_recap.png")
        page.screenshot(path=screenshot_path)
        print(f"Screenshot saved to {screenshot_path}")

        browser.close()

if __name__ == "__main__":
    verify_new_content()
