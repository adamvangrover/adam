from playwright.sync_api import sync_playwright

def verify_credit_memo(page):
    print("Verifying Credit Memo Automation...")
    url = "http://localhost:8085/showcase/credit_memo_automation.html"
    page.goto(url)

    # Wait for sidebar to populate (UniversalLoader init)
    page.wait_for_selector("#library-list div", timeout=5000)
    print("Sidebar populated.")

    # Click Apple Inc. (Assuming it's in the list or filtered)
    # The list items are divs with text.
    # We can search for text "Apple Inc."
    page.get_by_text("Apple Inc.").first.click()

    # Wait for content to load (the memo title inside the container, not the app header)
    # Header should appear with "APPLE INC."
    page.wait_for_selector("#memo-container h1", timeout=10000)

    # Take screenshot
    page.screenshot(path="verification/credit_memo_automation.png")
    print("Screenshot saved: verification/credit_memo_automation.png")

def verify_sovereign_dashboard(page):
    print("Verifying Sovereign Dashboard...")
    url = "http://localhost:8085/showcase/sovereign_dashboard.html"
    page.goto(url)

    # Wait for ticker list
    page.wait_for_selector("#tickerList li", timeout=5000)

    # Click AAPL
    page.get_by_text("AAPL").click()

    # Wait for Chart canvas
    page.wait_for_selector("#financialChart", timeout=5000)

    # Take screenshot
    page.screenshot(path="verification/sovereign_dashboard.png")
    print("Screenshot saved: verification/sovereign_dashboard.png")

def verify_credit_memo_v2(page):
    print("Verifying Credit Memo v2...")
    url = "http://localhost:8085/showcase/credit_memo_v2.html"
    page.goto(url)

    # Wait for dropdown to be populated (more than 1 option)
    page.wait_for_function("document.querySelectorAll('#borrower-select option').length > 1", timeout=5000)

    # Select Apple Inc. (We need to find the value)
    # UniversalLoader populates values as IDs (e.g. "Apple_Inc" or "AAPL")
    # Let's just select the second option (first is placeholder)
    page.select_option("#borrower-select", index=1)

    # Click Generate
    page.click("#generate-btn")

    # Wait for Memo Content
    page.wait_for_selector(".memo-paper", timeout=10000)

    # Take screenshot (scroll to tools)
    page.get_by_text("Tools").scroll_into_view_if_needed()
    page.screenshot(path="verification/credit_memo_v2.png")
    print("Screenshot saved: verification/credit_memo_v2.png")

def verify_credit_memo_orig(page):
    print("Verifying Credit Memo (Original)...")
    url = "http://localhost:8085/showcase/credit_memo.html"
    page.goto(url)

    # Wait for dropdown to be populated
    page.wait_for_function("document.querySelectorAll('#borrower-select option').length > 1", timeout=5000)

    # Select Apple Inc.
    page.select_option("#borrower-select", index=1)

    # Click Generate
    page.click("#generate-btn")

    # Wait for Memo Content
    page.wait_for_selector(".memo-paper", timeout=10000)

    # Take screenshot (scroll to tools)
    page.get_by_text("Tools").scroll_into_view_if_needed()
    page.screenshot(path="verification/credit_memo_original.png")
    print("Screenshot saved: verification/credit_memo_original.png")

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()

    try:
        verify_credit_memo(page)
        verify_sovereign_dashboard(page)
        verify_credit_memo_v2(page)
        verify_credit_memo_orig(page)
    except Exception as e:
        print(f"Verification failed: {e}")
    finally:
        browser.close()
