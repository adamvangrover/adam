
from playwright.sync_api import sync_playwright
import os
import time

def run():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context()
        page = context.new_page()

        # 1. Verify Evolution Page
        cwd = os.getcwd()
        url = f"file://{cwd}/showcase/evolution.html"
        print(f"Navigating to {url}")
        page.goto(url)
        time.sleep(1)
        page.screenshot(path="verification/evolution_page.png", full_page=True)
        print("Evolution page verified.")

        # 2. Verify Command Palette
        # Simulate Ctrl+K
        print("Testing Command Palette...")
        page.keyboard.press("Control+k")
        time.sleep(0.5)

        # Check if overlay is visible
        is_visible = page.is_visible("#cmd-palette-overlay")
        if is_visible:
            print("Command Palette is visible.")
            page.screenshot(path="verification/command_palette.png")

            # Type "deep"
            page.fill("#cmd-input", "deep")
            time.sleep(0.5)

            # Check results
            results = page.inner_text("#cmd-results")
            if "Deep Dive Analyst" in results:
                print("Search functionality verified (found 'Deep Dive Analyst').")
            else:
                print(f"Search failed. Results: {results}")
        else:
            print("Command Palette failed to open.")

        browser.close()

if __name__ == "__main__":
    run()
