from playwright.sync_api import sync_playwright
import time
import os

def run():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        # Log console messages
        page.on("console", lambda msg: print(f"CONSOLE [{msg.type}]: {msg.text}"))

        # Log unhandled exceptions
        page.on("pageerror", lambda err: print(f"PAGE ERROR: {err}"))

        print("Navigating to local server...")
        # Make sure our local python3 -m http.server 8000 is still running
        try:
            page.goto("http://localhost:8000/showcase/market_mayhem_graph.html", wait_until="networkidle", timeout=10000)

            # Wait a few seconds for data to load and 3D objects to render
            page.wait_for_timeout(3000)

            # Take screenshot
            screenshot_path = os.path.join(os.getcwd(), "verification/graph_success.png")
            page.screenshot(path=screenshot_path)
            print(f"Screenshot saved to {screenshot_path}")

        except Exception as e:
            print(f"Failed: {e}")
        finally:
            browser.close()

if __name__ == "__main__":
    run()
