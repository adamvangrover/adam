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

def unified_action(page):
    # Select the new scenario
    page.select_option("#scenario-select", "Reflationary Agentic Boom")
    time.sleep(1) # Wait for update

def brain_action(page):
    # Just wait for graph to settle
    time.sleep(5)

if __name__ == "__main__":
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.set_viewport_size({"width": 1280, "height": 720})

        try:
            # Verify Unified Banking
            verify(page, "http://localhost:8000/showcase/unified_banking.html", "unified_banking", unified_action)

            # Verify System Brain
            verify(page, "http://localhost:8000/showcase/system_brain.html", "system_brain", brain_action)

        except Exception as e:
            print(f"Error: {e}")
        finally:
            browser.close()
