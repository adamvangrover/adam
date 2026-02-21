import json
from playwright.sync_api import sync_playwright

def run():
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        context = browser.new_context()
        page = context.new_page()

        # Mock the API endpoint
        def handle_route(route):
            print("Intercepted request to:", route.request.url)
            route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps([
                    {"id": "1", "text": "Analyzing market data..."},
                    {"id": "2", "text": "Calculating risk metrics..."}
                ])
            )

        page.route("**/api/intercom/stream", handle_route)

        try:
            print("Navigating to http://localhost:3000")
            page.goto("http://localhost:3000")

            # Check if Agent Intercom is present
            print("Waiting for AGENT INTERCOM text...")
            page.wait_for_selector('text=AGENT INTERCOM', timeout=20000)

            # Check if thoughts are rendered
            print("Waiting for thought text...")
            page.wait_for_selector('text=Analyzing market data...', timeout=20000)

            # Screenshot
            page.screenshot(path="verification_screenshots/agent_intercom.png")
            print("Screenshot saved to verification_screenshots/agent_intercom.png")

        except Exception as e:
            print(f"Error: {e}")
            page.screenshot(path="verification_screenshots/error.png")
        finally:
            browser.close()

if __name__ == "__main__":
    run()
