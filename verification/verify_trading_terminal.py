from playwright.sync_api import sync_playwright
import time

def run():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        # Navigate to the Trading Terminal page
        # Assuming the dev server is running on localhost:3000
        # and hash router is used
        try:
            page.goto("http://localhost:3000/#/trading-terminal")

            # Wait for content to load
            page.wait_for_selector("text=ADAM v26.0 | Trading Terminal", timeout=10000)

            # Verify Algo Strategies section
            page.wait_for_selector("text=Algorithmic Strategies")
            page.wait_for_selector("text=Momentum Alpha")

            # Verify Robo Advisor section
            page.wait_for_selector("text=Robo-Advisor Allocation")
            page.wait_for_selector("text=Equities")

            # Take screenshot
            page.screenshot(path="verification/trading_terminal.png", full_page=True)
            print("Verification successful: Screenshot saved to verification/trading_terminal.png")

        except Exception as e:
            print(f"Verification failed: {e}")
            # Take screenshot even if failed to see state
            page.screenshot(path="verification/error_state.png", full_page=True)

        finally:
            browser.close()

if __name__ == "__main__":
    run()
