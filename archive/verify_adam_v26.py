import os
from playwright.sync_api import sync_playwright

def verify_showcase():
    file_path = f"file://{os.path.abspath('showcase/adam_v26.html')}"
    print(f"Loading {file_path}...")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1280, "height": 800})

        # Navigate and wait for particles & animations to settle
        page.goto(file_path, wait_until="networkidle")
        page.wait_for_timeout(2000)

        screenshot_path = "verification_adam_v26.png"
        page.screenshot(path=screenshot_path, full_page=True)
        print(f"Saved full page screenshot to {screenshot_path}")

        browser.close()

if __name__ == "__main__":
    verify_showcase()
