from playwright.sync_api import sync_playwright

def verify_archive_update():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        # Navigate to Archive
        page.goto("http://localhost:8000/market_mayhem_archive.html")
        page.wait_for_load_state("networkidle")

        # Search for the new entry
        item = page.locator(".archive-item").filter(has_text="MARKET MAYHEM: The Weekly Briefing").first
        if not item.is_visible():
            print("FAILED: Could not find the new archive item.")
            browser.close()
            return

        print("SUCCESS: Found the new archive item.")

        # Click the access button within that item
        item.get_by_role("link", name="ACCESS").click()

        # Wait for navigation
        page.wait_for_load_state("networkidle")

        # Verify content of the new page
        title = page.title()
        expected_title = "ADAM v24.1 :: MARKET MAYHEM: The Weekly Briefing"
        if title != expected_title:
            print(f"FAILED: Title mismatch. Expected '{expected_title}', got '{title}'")
        else:
            print("SUCCESS: Title matches.")

        # Check for specific content
        if page.get_by_text("The Dow has finally punched through the psychological 50,000 barrier").count() > 0:
             print("SUCCESS: Content verified.")
        else:
             print("FAILED: Content mismatch.")

        # Screenshot
        page.screenshot(path="verification/archive_verification_manual.png", full_page=True)
        print("Screenshot saved to verification/archive_verification_manual.png")

        browser.close()

if __name__ == "__main__":
    verify_archive_update()
