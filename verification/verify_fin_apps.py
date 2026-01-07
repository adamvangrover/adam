from playwright.sync_api import sync_playwright
import os

def run():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        # Verify DCF Update
        dcf_path = os.path.abspath("showcase/apps/dcf.html")
        dcf_url = f"file://{dcf_path}"
        print(f"Navigating to {dcf_url}")
        page.goto(dcf_url)
        page.wait_for_selector("table")
        page.screenshot(path="verification/dcf_refined.png")
        print("DCF Refined screenshot saved.")

        # Verify Capital Stack
        cap_path = os.path.abspath("showcase/apps/capital_stack.html")
        cap_url = f"file://{cap_path}"
        print(f"Navigating to {cap_url}")
        page.goto(cap_url)
        page.wait_for_selector("#chart-stack")
        page.screenshot(path="verification/capital_stack.png")
        print("Capital Stack screenshot saved.")

        browser.close()

if __name__ == "__main__":
    run()
