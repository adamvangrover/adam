import time
from playwright.sync_api import sync_playwright

def test_credit_advanced():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context()
        page = context.new_page()
        page.on("console", lambda msg: print(f"CONSOLE: {msg.text}"))

        print("Navigating to credit_memo_v2.html...")
        page.goto("http://127.0.0.1:8091/showcase/credit_memo_v2.html")

        # Check Title
        print(f"DEBUG: Actual Title is '{page.title()}'")
        assert "ADAM | Enterprise Credit Analysis" in page.title()
        print("PASS: Page loaded.")

        # Check Tools
        assert page.locator("#upload-btn").is_visible()
        assert page.locator("#download-btn").is_visible()
        print("PASS: Upload/Download buttons visible.")

        # Select Apple Inc.
        print("Selecting Apple Inc...")
        # Debug options
        options = page.eval_on_selector_all("#borrower-select option", "opts => opts.map(o => o.text)")
        print(f"DEBUG: Options are {options}")

        page.select_option("#borrower-select", label="Apple Inc. (Risk: 75)")
        page.click("#generate-btn")

        # Wait for Analysis
        print("Waiting for analysis to complete...")
        try:
            page.wait_for_selector(".memo-paper", timeout=30000)
        except Exception as e:
            print("ERROR: Timeout waiting for memo paper.")
            terminal_text = page.locator("#agent-terminal").text_content()
            print(f"TERMINAL LOGS:\n{terminal_text}")
            page.screenshot(path="verification_screenshots/error_state.png")
            raise e

        # Check Outlook Section (New)
        # Target specific Outlook Conviction
        outlook = page.locator(".memo-paper div").filter(has_text="CONVICTION").first
        assert outlook.is_visible()

        rating = page.locator("text=STRONG BUY").or_(page.locator("text=BUY")).or_(page.locator("text=HOLD"))
        assert rating.first.is_visible()
        print("PASS: Analyst Outlook section visible.")

        # Check Sensitivity Matrix (New)
        matrix = page.locator("h2:has-text('Sensitivity')").or_(page.locator("h3:has-text('Sensitivity')"))
        assert matrix.first.is_visible()
        assert page.locator("table").is_visible()
        print("PASS: Sensitivity Matrix visible.")

        # Check Chart
        chart = page.locator("#finChart")
        assert chart.is_visible()
        print("PASS: Chart visible.")

        # Test Download (Mock)
        print("Testing Download...")
        with page.expect_download() as download_info:
            page.click("#download-btn")
        download = download_info.value
        path = download.path()
        print(f"PASS: Download triggered. Saved to {path}")

        # Screenshot
        page.screenshot(path="verification_screenshots/credit_memo_advanced.png")
        print("Screenshot saved.")

        browser.close()

if __name__ == "__main__":
    test_credit_advanced()
