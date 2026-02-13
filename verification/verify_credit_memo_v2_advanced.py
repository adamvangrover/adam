import time
from playwright.sync_api import sync_playwright

def run():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        page.goto("http://localhost:8085/credit_memo_v2.html")

        select = page.locator("#borrower-select")
        select.wait_for()
        page.wait_for_function("document.getElementById('borrower-select').options.length > 1")
        select.select_option(value="credit_memo_Apple_Inc.json")
        page.locator("#generate-btn").click()

        page.locator("#memo-content").wait_for(state="visible", timeout=10000)

        # Edit Mode
        page.locator("#edit-btn").click()
        title = page.locator("h1.editable-content")
        assert title.get_attribute("contenteditable") == "true"

        # Interactive DCF
        wacc_input = page.locator("#dcf-wacc")
        wacc_input.scroll_into_view_if_needed()
        initial_price = page.locator("#dcf-share-price").inner_text()
        wacc_input.fill("12.0")
        wacc_input.evaluate("el => el.dispatchEvent(new Event('change'))")

        # Risk Quant
        quant_panel = page.locator("#risk-quant-panel")
        print(f"Quant Panel Content: {quant_panel.inner_text()}")

        # Wait a bit for JS to populate if it was lagging (unlikely given synchronous call order)
        time.sleep(1)
        print(f"Quant Panel Content after sleep: {quant_panel.inner_text()}")

        assert "CREDIT RISK MODEL" in quant_panel.inner_text()

        page.screenshot(path="verification/credit_memo_v2_advanced.png", full_page=True)
        browser.close()

if __name__ == "__main__":
    run()
