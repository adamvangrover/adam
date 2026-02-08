from playwright.sync_api import sync_playwright
import time
import json

def run():
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()

        # Mock the API
        mock_thoughts_1 = [{"id": "1", "text": "Analyzing market volatility..."}]
        mock_thoughts_2 = [{"id": "1", "text": "Analyzing market volatility..."}, {"id": "2", "text": "Risk detected."}]

        # Use a closure or global to cycle responses?
        # Playwright route handler can be dynamic.

        call_count = 0

        def handle_route(route):
            nonlocal call_count
            call_count += 1
            print(f"Intercepted API call #{call_count}")

            if call_count <= 1:
                route.fulfill(status=200, body=json.dumps(mock_thoughts_1))
            else:
                route.fulfill(status=200, body=json.dumps(mock_thoughts_2))

        page.route("**/api/intercom/stream", handle_route)

        print("Navigating to Dashboard...")
        page.goto("http://localhost:3000/#/")

        # Wait for Agent Intercom
        try:
            page.wait_for_selector("text=AGENT INTERCOM", timeout=30000)
            print("Found Agent Intercom.")
        except Exception as e:
            print(f"Failed to find Agent Intercom: {e}")
            page.screenshot(path="verification/intercom_error.png")
            return

        # Wait for first thought
        try:
            page.wait_for_selector("text=Analyzing market volatility...", timeout=10000)
            print("Found first thought.")
        except Exception as e:
            print(f"Failed to find first thought: {e}")
            page.screenshot(path="verification/intercom_failed_1.png")
            return

        # Wait for polling (2s interval + some buffer)
        print("Waiting for polling cycle...")
        time.sleep(3)

        # Wait for second thought
        try:
            page.wait_for_selector("text=Risk detected.", timeout=10000)
            print("Found second thought (polling works).")
        except Exception as e:
            print(f"Failed to find second thought: {e}")
            page.screenshot(path="verification/intercom_failed_2.png")
            return

        page.screenshot(path="verification/intercom_success.png")
        print("Success screenshot taken.")

        browser.close()

if __name__ == "__main__":
    run()
