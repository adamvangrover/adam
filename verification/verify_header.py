from playwright.sync_api import sync_playwright, expect

def verify_header_aria_labels():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        try:
            # Increase timeout to 60s
            page.goto("http://localhost:3000", timeout=60000)

            # Wait for header to load
            page.wait_for_selector("header")

            # Check Menu Button
            menu_btn = page.get_by_label("Toggle navigation menu")
            expect(menu_btn).to_be_visible()

            # Check Bell Button
            bell_btn = page.get_by_label("View notifications")
            expect(bell_btn).to_be_visible()

            # Take screenshot
            page.screenshot(path="verification/header_verification.png")
            print("Verification successful: ARIA labels found.")

        except Exception as e:
            print(f"Verification failed: {e}")

        finally:
            browser.close()

if __name__ == "__main__":
    verify_header_aria_labels()
