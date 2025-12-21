
from playwright.sync_api import sync_playwright


def verify_nav():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        # Verify Root Index
        print("Verifying Root Index...")
        page.goto("http://localhost:8000/index.html")
        # Wait for nav to be injected (nav.js runs on DOMContentLoaded)
        try:
            page.wait_for_selector("#side-nav", timeout=5000)
            print("Nav found on Root Index.")
        except:
            print("Nav NOT found on Root Index.")

        page.screenshot(path="verification/root_index.png")
        print("Root Index screenshot taken.")

        # Verify Showcase Index
        print("Verifying Showcase Index...")
        page.goto("http://localhost:8000/showcase/index.html")
        try:
            page.wait_for_selector("#side-nav", timeout=5000)
            print("Nav found on Showcase Index.")
        except:
            print("Nav NOT found on Showcase Index.")

        page.screenshot(path="verification/showcase_index.png")
        print("Showcase Index screenshot taken.")

        browser.close()

if __name__ == "__main__":
    verify_nav()
