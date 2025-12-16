from playwright.sync_api import sync_playwright, expect

def test_search_ux():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        try:
            # 1. Navigate to the app
            page.goto("http://localhost:3000")

            # 2. Check for the input with the new ARIA label
            search_input = page.get_by_label("Search system knowledge")
            expect(search_input).to_be_visible()

            # 3. Test Keyboard Shortcut (Ctrl+K)
            # Click somewhere else first to ensure blur
            page.locator("body").click()
            page.keyboard.press("Control+k")

            # Verify focus
            expect(search_input).to_be_focused()
            print("Ctrl+K focus verified")

            # 4. Type text and verify Clear button appears
            search_input.fill("Test Query")

            clear_button = page.get_by_label("Clear search")
            expect(clear_button).to_be_visible()
            print("Clear button visibility verified")

            # Take screenshot of the search state
            page.screenshot(path="verification/search_active.png")

            # 5. Test Clear button
            clear_button.click()
            expect(search_input).to_have_value("")
            # Also verify focus returns to input (UX enhancement)
            expect(search_input).to_be_focused()
            print("Clear button functionality verified")

        except Exception as e:
            print(f"Test failed: {e}")
            page.screenshot(path="verification/failure.png")
            raise e
        finally:
            browser.close()

if __name__ == "__main__":
    test_search_ux()
