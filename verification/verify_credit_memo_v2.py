import time
from playwright.sync_api import sync_playwright

def run():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        # Navigate to the page served by http.server
        page.goto("http://localhost:8085/credit_memo_v2.html")

        # 1. Verify Title
        print(f"Title: {page.title()}")
        assert "ADAM | Enterprise Credit Analysis" in page.title()

        # 2. Verify Dropdown population
        select = page.locator("#borrower-select")
        select.wait_for()

        # Wait for options to load (fetch is async)
        page.wait_for_function("document.getElementById('borrower-select').options.length > 1")

        options = select.locator("option").all_inner_texts()
        print(f"Options found: {len(options)}")
        assert len(options) > 1

        # 3. Select Apple Inc.
        # Find option value for Apple
        # We know from previous `read_file` that Apple is likely `credit_memo_Apple_Inc.json`
        select.select_option(value="credit_memo_Apple_Inc.json")

        # 4. Click Execute Analysis
        btn = page.locator("#generate-btn")
        btn.click()

        # 5. Wait for content
        # The button changes text to "GENERATING..." then back
        # Content #memo-content should become visible
        memo_content = page.locator("#memo-content")
        memo_content.wait_for(state="visible", timeout=10000)

        # Verify header text
        header = memo_content.locator("h1")
        print(f"Memo Header: {header.inner_text()}")
        assert "Apple Inc." in header.inner_text()

        # 6. Click a citation
        citation = page.locator(".citation-pin").first
        if citation.count() > 0:
            print("Clicking citation...")
            citation.click()
            # Verify evidence viewer update
            viewer_label = page.locator("#doc-id-label")
            # Wait for text to not be [NO DOC]
            page.wait_for_function("document.getElementById('doc-id-label').innerText !== '[NO DOC]'")
            print(f"Evidence Label: {viewer_label.inner_text()}")

        # 7. Screenshot
        page.screenshot(path="verification/credit_memo_v2_screenshot.png", full_page=True)
        print("Screenshot saved to verification/credit_memo_v2_screenshot.png")

        browser.close()

if __name__ == "__main__":
    run()
