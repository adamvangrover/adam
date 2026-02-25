from playwright.sync_api import sync_playwright
import time

def run():
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()

        # Capture console logs
        page.on("console", lambda msg: print(f"CONSOLE: {msg.text}"))
        page.on("pageerror", lambda msg: print(f"PAGE ERROR: {msg}"))

        try:
            print("Navigating...")
            response = page.goto("http://localhost:8000/showcase/market_mayhem_graph.html")
            print(f"Status: {response.status}")

            # Wait for canvas
            print("Waiting for canvas...")
            page.wait_for_selector("#canvas-container canvas", timeout=20000)

            # Wait a bit for initialization and animation
            print("Waiting for animation...")
            page.wait_for_timeout(3000)

            # Simulate mouse move to trigger interaction logic (updateInteraction)
            print("Moving mouse...")
            page.mouse.move(500, 500)
            page.wait_for_timeout(500)

            # Simulate click
            print("Clicking...")
            page.mouse.click(500, 500)
            page.wait_for_timeout(500)

            # Take screenshot
            page.screenshot(path="verification_graph.png")
            print("Screenshot saved to verification_graph.png")

        except Exception as e:
            print(f"Error: {e}")

        finally:
            browser.close()

if __name__ == "__main__":
    run()
