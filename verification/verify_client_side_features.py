from playwright.sync_api import sync_playwright

def verify_buttons(page, url, button_texts):
    print(f"Verifying {url}...")
    page.goto(url)
    page.wait_for_load_state("networkidle")

    # Wait for UniversalLoader to init and likely inject buttons
    page.wait_for_timeout(2000)

    for text in button_texts:
        # Case insensitive search for button text or icon class if needed
        # We look for button that contains the text
        if "Edit" in text and "credit_memo_automation" in url:
             # Automation page has "Edit" injected
             found = page.get_by_role("button", name="Edit").is_visible()
        elif "JSON" in text:
             # Check for Download JSON or similar
             found = page.get_by_text(text, exact=False).first.is_visible()
        elif "Load" in text or "UPLOAD" in text:
             found = page.get_by_text(text, exact=False).first.is_visible()
        else:
             found = page.get_by_text(text).first.is_visible()

        if found:
            print(f"  [PASS] Found button: {text}")
        else:
            print(f"  [FAIL] Missing button: {text}")
            # page.screenshot(path=f"verification/fail_{text.replace(' ','_')}.png")

def verify_client_features():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        # 1. Credit Memo v2
        verify_buttons(page, "http://localhost:8085/showcase/credit_memo_v2.html", ["EDIT MODE", "JSON", "LOAD"])

        # 2. Credit Memo Automation (v1)
        verify_buttons(page, "http://localhost:8085/showcase/credit_memo_automation.html", ["Edit", "Save JSON", "Load"])

        # 3. Sovereign Dashboard
        verify_buttons(page, "http://localhost:8085/showcase/sovereign_dashboard.html", ["UPLOAD JSON", "DOWNLOAD JSON"])

        browser.close()

if __name__ == "__main__":
    verify_client_features()
