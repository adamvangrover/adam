
from playwright.sync_api import sync_playwright

def verify_empty_state_intercept():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        # Intercept the manifest request to return empty agents
        def handle_manifest_request(route):
            route.fulfill(
                status=200,
                content_type="application/json",
                body='{"agents": [], "reports": []}'
            )

        # Monitor networks requests
        page.route("**/manifest.json", handle_manifest_request) # Original static file
        page.route("**/api/manifest", handle_manifest_request) # API call

        # Navigate to the agents page
        page.goto("http://localhost:4173/agents")

        # Wait for the empty state to appear
        try:
            page.wait_for_selector("text=NO AGENTS DETECTED", timeout=10000)
            print("Successfully found 'NO AGENTS DETECTED' text.")
        except Exception as e:
            print(f"Failed to find text: {e}")
            page.screenshot(path="verification/agent_registry_failed.png")
            return

        # Take a screenshot
        page.screenshot(path="verification/agent_registry_empty_state.png")

        browser.close()
        print("Verification screenshot saved to verification/agent_registry_empty_state.png")

if __name__ == "__main__":
    verify_empty_state_intercept()
