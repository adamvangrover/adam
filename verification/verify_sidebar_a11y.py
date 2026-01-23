import time
from playwright.sync_api import sync_playwright, expect

def verify_sidebar_a11y():
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        try:
            # Wait for server to start
            time.sleep(5)

            # Navigate to the app
            page.goto("http://localhost:5173/")

            # Wait for sidebar to be visible
            sidebar = page.locator("aside")
            expect(sidebar).to_be_visible()

            # Check for the progressbar role within the sidebar
            progressbar = sidebar.get_by_role("progressbar", name="SYSTEM RESOURCE")
            expect(progressbar).to_be_visible()

            # Verify attributes
            valuenow = progressbar.get_attribute("aria-valuenow")
            valuemin = progressbar.get_attribute("aria-valuemin")
            valuemax = progressbar.get_attribute("aria-valuemax")
            labelledby = progressbar.get_attribute("aria-labelledby")

            print(f"aria-valuenow: {valuenow}")
            print(f"aria-valuemin: {valuemin}")
            print(f"aria-valuemax: {valuemax}")
            print(f"aria-labelledby: {labelledby}")

            if valuenow == "65" and valuemin == "0" and valuemax == "100" and labelledby == "resource-usage-label":
                print("SUCCESS: All accessibility attributes present and correct.")
            else:
                print("FAILURE: Attributes missing or incorrect.")

            # Take screenshot of the sidebar area
            sidebar.screenshot(path="verification/sidebar_verification.png")

        except Exception as e:
            print(f"Error: {e}")
        finally:
            browser.close()

if __name__ == "__main__":
    verify_sidebar_a11y()
