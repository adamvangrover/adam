from playwright.sync_api import sync_playwright

def verify_frontend():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        # Navigate to the served page
        page.goto("http://localhost:8000/quantum_search.html")

        # Wait for data to load
        try:
            # Wait for odds display
            page.wait_for_selector("#odds-display", state="visible", timeout=5000)

            # Wait for list items (updated selector)
            page.wait_for_selector("#raw-candidates-list > div", state="visible", timeout=5000)

            # Wait for the chart
            page.wait_for_selector("#scheduleChart", state="visible", timeout=5000)

            # Take screenshot
            page.screenshot(path="verification/quantum_search.png", full_page=True)
            print("Screenshot taken: verification/quantum_search.png")

        except Exception as e:
            print(f"Error: {e}")
            page.screenshot(path="verification/error.png")

        browser.close()

if __name__ == "__main__":
    verify_frontend()
