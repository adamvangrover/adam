from playwright.sync_api import sync_playwright
import os

def run():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        # Point to the local file
        file_path = os.path.abspath("showcase/monte_carlo_visualizer.html")
        page.goto(f"file://{file_path}")

        # Wait for chart to render (it has animation duration 2000ms)
        page.wait_for_timeout(2500)

        # Screenshot
        screenshot_path = "verification/monte_carlo_viz.png"
        page.screenshot(path=screenshot_path, full_page=True)
        print(f"Screenshot saved to {screenshot_path}")

        browser.close()

if __name__ == "__main__":
    run()
