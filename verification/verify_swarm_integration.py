from playwright.sync_api import sync_playwright, expect

def verify_swarm_integration():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        try:
            print("Navigating to Prompt Alpha...")
            page.goto("http://localhost:3000/#/prompt-alpha")
            page.wait_for_load_state("networkidle")

            # 1. Verify Title and User Stats
            print("Verifying Header...")
            # Use role to be more specific
            expect(page.get_by_role("heading", name="PROMPT ALPHA")).to_be_visible()
            expect(page.get_by_text("OPERATOR RANK")).to_be_visible()

            # 2. Verify Swarm Navigation
            print("Navigating to SWARM view...")
            page.get_by_role("button", name="SWARM").click()

            # Wait for grid to appear
            expect(page.get_by_text("Active Agents")).to_be_visible()
            expect(page.get_by_text("Network Load")).to_be_visible()

            # Take screenshot of Swarm
            print("Capturing Swarm Screenshot...")
            page.screenshot(path="/home/jules/verification/swarm_view.png")

            # 3. Verify News Navigation
            print("Navigating to NEWS view...")
            page.get_by_role("button", name="NEWS", exact=True).click()

            expect(page.get_by_text("NEWS WIRE")).to_be_visible()

            # Take screenshot of News
            print("Capturing News Screenshot...")
            page.screenshot(path="/home/jules/verification/news_view.png")

            print("Verification Successful!")

        except Exception as e:
            print(f"Verification Failed: {e}")
            page.screenshot(path="/home/jules/verification/error.png")
            raise e
        finally:
            browser.close()

if __name__ == "__main__":
    verify_swarm_integration()
