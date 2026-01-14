from playwright.sync_api import sync_playwright
import time
import os

def verify_dashboard():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        # Navigate to Index via localhost
        # We assume the server is running on port 8000
        print("Navigating to Index...")
        try:
            page.goto("http://localhost:8000/showcase/index.html", timeout=10000)
        except Exception as e:
            print(f"Failed to load page: {e}")
            return

        # Click the link
        print("Clicking Launch Twin...")
        try:
            with page.expect_navigation(timeout=5000):
                page.get_by_text("LAUNCH TWIN").click()
        except Exception as e:
             print(f"Navigation failed or timed out: {e}")
             # Try direct navigation
             print("Attempting direct navigation to unified_banking.html...")
             page.goto("http://localhost:8000/showcase/unified_banking.html")

        print("Navigated to Unified Banking Dashboard.")

        # Wait for network graph to render (it's canvas based)
        time.sleep(3)

        # Check title
        title = page.title()
        print(f"Page Title: {title}")

        # Take screenshot
        output_path = "verification/dashboard_screenshot.png"
        page.screenshot(path=output_path)
        print(f"Screenshot saved to {output_path}")

        browser.close()

if __name__ == "__main__":
    verify_dashboard()
