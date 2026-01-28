from playwright.sync_api import sync_playwright, expect

def verify_intercom():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        try:
            print("Navigating to http://localhost:3000...")
            page.goto("http://localhost:3000")
            # Wait for the network to be idle, but don't fail if some long polling keeps it active
            try:
                page.wait_for_load_state("networkidle", timeout=5000)
            except:
                print("Network didn't idle (expected due to polling), proceeding...")

            # The Intercom starts open (isCollapsed=false)
            # So we expect the minimize button to be visible.
            minimize_btn = page.get_by_role("button", name="Minimize Agent Intercom")
            expect(minimize_btn).to_be_visible(timeout=10000)
            print("Confirmed: Intercom starts OPEN.")

            # Check for the feed region
            feed = page.get_by_role("log", name="Agent Thoughts Feed")
            expect(feed).to_be_visible()
            print("Confirmed: Feed region exists and is visible.")

            # Take screenshot of open state
            page.screenshot(path="verification/intercom_open.png")
            print("Screenshot saved: verification/intercom_open.png")

            # Click minimize
            minimize_btn.click()
            print("Clicked Minimize button.")

            # Expect launcher button to appear
            launcher_btn = page.get_by_role("button", name="Open Agent Intercom")
            expect(launcher_btn).to_be_visible()
            print("Confirmed: Launcher button appeared.")

            # Take screenshot of closed state
            page.screenshot(path="verification/intercom_closed.png")
            print("Screenshot saved: verification/intercom_closed.png")

            # Click launcher to reopen
            launcher_btn.click()
            print("Clicked Launcher button.")

            # Expect minimize button again
            expect(minimize_btn).to_be_visible()
            print("Confirmed: Intercom reopened.")

        except Exception as e:
            print(f"Test failed: {e}")
            page.screenshot(path="verification/failure.png")
        finally:
            browser.close()

if __name__ == "__main__":
    verify_intercom()
