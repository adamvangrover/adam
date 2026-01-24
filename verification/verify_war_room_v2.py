from playwright.sync_api import sync_playwright
import os
import time

def verify_war_room():
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()

        # Absolute path to the file
        cwd = os.getcwd()
        file_path = f"file://{cwd}/showcase/war_room_v2.html"

        print(f"Navigating to {file_path}")
        page.goto(file_path)

        # Wait for the scan line animation or initial render
        time.sleep(2)

        # Take screenshot
        output_path = "verification/war_room_v2.png"
        page.screenshot(path=output_path, full_page=True)
        print(f"Screenshot saved to {output_path}")

        browser.close()

if __name__ == "__main__":
    verify_war_room()
