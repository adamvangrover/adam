from playwright.sync_api import sync_playwright
import time

def verify_deployment(page):
    # Navigate to Mission Control
    page.goto("http://localhost:8080/index.html")

    # Wait for the "App Deploy" link to be visible and click it
    page.wait_for_selector("text=App Deploy")
    page.click("text=App Deploy")

    # Wait for navigation
    page.wait_for_load_state("networkidle")
    time.sleep(2) # Give JS time to render the tree

    # Take screenshot of the whole page
    page.screenshot(path="verification/deployment_ui.png")
    print("Screenshot taken")

if __name__ == "__main__":
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        try:
            verify_deployment(page)
        except Exception as e:
            print(f"Verification failed: {e}")
            page.screenshot(path="verification/error.png")
        finally:
            browser.close()
