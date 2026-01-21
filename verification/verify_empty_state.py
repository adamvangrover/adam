
from playwright.sync_api import sync_playwright

def verify_empty_state():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        # Navigate to the agents page
        page.goto("http://localhost:4173/agents")

        # Wait for the empty state to appear (it should replace the loading state)
        # We look for the text "NO AGENTS DETECTED"
        page.wait_for_selector("text=NO AGENTS DETECTED")

        # Take a screenshot
        page.screenshot(path="verification/agent_registry_empty_state.png")

        browser.close()
        print("Verification screenshot saved to verification/agent_registry_empty_state.png")

if __name__ == "__main__":
    verify_empty_state()
