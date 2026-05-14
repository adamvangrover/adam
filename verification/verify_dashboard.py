from playwright.sync_api import sync_playwright
import time

def verify_dashboard():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        try:
            print("Navigating to HTML Showcases...")
            page.goto("file:///app/showcase/credit_valuation_architect.html")
            time.sleep(2)
            page.screenshot(path="/app/verification/cva.png", full_page=True)
            print("Screenshot saved to /app/verification/cva.png")

            page.goto("file:///app/showcase/credit_architect.html")
            time.sleep(2)
            page.screenshot(path="/app/verification/ca.png", full_page=True)
            print("Screenshot saved to /app/verification/ca.png")

        except Exception as e:
            print(f"Error: {e}")
        finally:
            browser.close()

if __name__ == "__main__":
    verify_dashboard()
