from playwright.sync_api import sync_playwright

def test_dashboard():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        # Access via localhost to bypass CORS
        page.goto("http://localhost:8000/showcase/unified_banking_dashboard.html")

        # Wait for data to load
        try:
            page.wait_for_selector("#priceChart")
            # Wait for specific data element to ensure JS executed
            page.wait_for_selector("#orderLog div", timeout=5000)
        except Exception as e:
            print(f"Error waiting for selectors: {e}")

        # Take screenshot
        page.screenshot(path="verification/dashboard_screenshot_fixed.png", full_page=True)
        browser.close()

if __name__ == "__main__":
    test_dashboard()
