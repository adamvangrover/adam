from playwright.sync_api import sync_playwright
import os
import time

def verify_analyst_os():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context()
        page = context.new_page()

        # Path to analyst_os.html
        # Since I'm running from repo root, I can use relative path converted to absolute
        file_path = os.path.abspath("showcase/analyst_os.html")
        page.goto(f"file://{file_path}")

        # 1. Verify Intro Modal appears
        print("Checking Intro Modal...")
        page.wait_for_selector("#intro-modal")
        page.screenshot(path="verification/1_intro_modal.png")
        print("Captured intro modal.")

        # 2. Click Initialize (Close Modal)
        print("Closing Modal...")
        page.click("button:has-text('INITIALIZE WORKBENCH')")
        time.sleep(1) # Wait for fade out/animation

        # 3. Verify Desktop and Dock
        print("Checking Desktop...")
        page.screenshot(path="verification/2_desktop_empty.png")

        # 4. Open DCF App from Dock
        print("Opening DCF App...")
        page.click(".dock-item[data-title='DCF Valuator']")
        time.sleep(1)
        page.screenshot(path="verification/3_dcf_window.png")

        # 5. Open Market Data App from Dock
        print("Opening Market Data App...")
        page.click(".dock-item[data-title='Market Data']")
        time.sleep(1)
        page.screenshot(path="verification/4_market_window.png")

        browser.close()

if __name__ == "__main__":
    try:
        verify_analyst_os()
        print("Verification script finished successfully.")
    except Exception as e:
        print(f"Verification failed: {e}")
