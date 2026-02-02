from playwright.sync_api import sync_playwright
import os

def test_crisis_response_ui():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        # Load the local HTML file directly
        file_path = os.path.abspath("showcase/crisis_response.html")
        page.goto(f"file://{file_path}")

        # Wait for the page to load
        page.wait_for_selector("#analyzeBtn")

        # Take a screenshot of the initial state
        page.screenshot(path="verification/initial_state.png")
        print("Initial state screenshot captured.")

        # Simulate filling the inputs
        # Note: We can't easily mock the fetch call in a file:// protocol context without more complex setup,
        # but we can verify the UI elements exist.

        # Verify "Data Integrity & Verification" panel exists
        if page.locator("#integrityPanel").count() > 0:
            print("PASS: Integrity Panel found.")
        else:
            print("FAIL: Integrity Panel not found.")

        # Verify "System Critique" panel exists (hidden by default)
        if page.locator("#critiquePanel").count() > 0:
            print("PASS: Critique Panel found.")
        else:
            print("FAIL: Critique Panel not found.")

        # Verify "Acceptance Meta" container exists
        if page.locator("#acceptanceMeta").count() > 0:
            print("PASS: Acceptance Meta found.")
        else:
            print("FAIL: Acceptance Meta not found.")

        browser.close()

if __name__ == "__main__":
    test_crisis_response_ui()
