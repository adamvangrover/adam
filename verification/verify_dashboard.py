from playwright.sync_api import sync_playwright, expect

def verify_dashboard():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        try:
            # Increase timeout to 60s
            page.goto("http://localhost:3000", timeout=60000)

            # Wait for dashboard to load (look for "ADAM v23.5")
            page.wait_for_selector("text=ADAM v23.5")

            # Take screenshot
            page.screenshot(path="verification/dashboard.png")
            print("Verification successful: Dashboard loaded.")

        except Exception as e:
            print(f"Verification failed: {e}")
            page.screenshot(path="verification/error.png")

        finally:
            browser.close()

if __name__ == "__main__":
    verify_dashboard()
