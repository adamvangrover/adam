import os
from playwright.sync_api import sync_playwright

def verify_war_room():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        # Load the local HTML file
        cwd = os.getcwd()
        file_url = f"file://{cwd}/showcase/war_room_v2.html"
        print(f"Loading: {file_url}")

        page.goto(file_url)

        # Wait for elements to appear (Simulating load)
        page.wait_for_selector('#ticker-feed')
        page.wait_for_selector('#riskCanvas')

        # Check title
        title = page.title()
        print(f"Title: {title}")
        assert "WAR ROOM" in title

        # Take screenshot
        output_dir = "/home/jules/verification"
        os.makedirs(output_dir, exist_ok=True)
        screenshot_path = os.path.join(output_dir, "war_room.png")
        page.screenshot(path=screenshot_path)
        print(f"Screenshot saved to {screenshot_path}")

        browser.close()

if __name__ == "__main__":
    verify_war_room()
