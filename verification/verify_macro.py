from playwright.sync_api import sync_playwright, expect
import time
import os

def verify(page, url, name, action=None):
    print(f"Navigating to {url}...")
    page.goto(url)

    if action:
        action(page)

    # Wait for visualizations to render
    time.sleep(3)

    screenshot_path = f"/home/jules/verification/{name}.png"
    os.makedirs(os.path.dirname(screenshot_path), exist_ok=True)
    page.screenshot(path=screenshot_path)
    print(f"Screenshot saved to {screenshot_path}")

def macro_action(page):
    # Just wait for chart
    time.sleep(2)

if __name__ == "__main__":
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.set_viewport_size({"width": 1280, "height": 720})

        try:
            # Verify Macro Strategy
            verify(page, "http://localhost:8000/showcase/macro_strategy.html", "macro_strategy", macro_action)

        except Exception as e:
            print(f"Error: {e}")
        finally:
            browser.close()
