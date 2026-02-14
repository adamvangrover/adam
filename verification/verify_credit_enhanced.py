import os
import time
from playwright.sync_api import sync_playwright

def verify_enhanced_credit_memo():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        # 1. Load Page
        page.on("console", lambda msg: print(f"BROWSER CONSOLE: {msg.text}"))
        page.on("pageerror", lambda err: print(f"BROWSER ERROR: {err}"))
        page.goto("http://localhost:8080/showcase/credit_memo_automation.html")
        page.wait_for_load_state("networkidle")

        # 2. Click on "Apple Inc." in library
        # We need to wait for library list to populate
        page.wait_for_selector("#library-list div")
        apple_item = page.get_by_text("Apple Inc.")
        if apple_item.count() > 0:
            apple_item.first.click()
        else:
            # Fallback: Click first item
            page.locator("#library-list div").first.click()

        print("Clicked item, waiting for generation...")

        # 3. Wait for Generation
        # Wait for log message indicating completion
        try:
            page.wait_for_selector("#progress-container", state="visible", timeout=2000)
        except:
            pass # Maybe missed it

        page.wait_for_selector("#progress-container", state="hidden", timeout=30000)

        # Ensure memo is rendered
        page.wait_for_selector("#memo-container h1", timeout=5000)

        # 4. Verify Financials
        print("Checking Financials...")
        page.click("#btn-tab-annex-a")
        page.wait_for_timeout(500) # Animation
        page.screenshot(path="verification/financials_enhanced.png")

        # Check for new fields
        content = page.content()
        if "Gross Debt" in content and "Financial Adjustments (Client-Side)" in content:
            print("PASS: Extended Financials and Adjustments Panel visible.")
        else:
            print("FAIL: Financials missing new fields.")

        # 5. Verify Valuation (Peer Comps)
        print("Checking Valuation...")
        page.click("#btn-tab-annex-b")
        page.wait_for_timeout(500)
        page.screenshot(path="verification/valuation_enhanced.png")

        content = page.content()
        if "Comparable Companies" in content and "EV/EBITDA" in content:
            print("PASS: Peer Comps visible.")
        else:
            print("FAIL: Peer Comps missing.")

        # 6. Verify System 2
        print("Checking System 2...")
        page.click("#btn-tab-system-2")
        page.wait_for_timeout(500)
        page.screenshot(path="verification/system2_enhanced.png")

        content = page.content()
        if "Quantitative Checks" in content and "Qualitative Signals" in content:
            print("PASS: System 2 Enhanced Analysis visible.")
        else:
            print("FAIL: System 2 missing enhanced sections.")

        # 7. Verify Agent Log
        print("Checking Agent Log...")
        audit_content = page.locator("#audit-table-body").inner_text()
        if "ArchivistAgent" in audit_content and "QuantAgent" in audit_content:
             print("PASS: Agent Log populated.")
        else:
             print("FAIL: Agent Log empty or missing agents. Content: " + audit_content[:100])

        browser.close()

if __name__ == "__main__":
    os.makedirs("verification", exist_ok=True)
    try:
        verify_enhanced_credit_memo()
    except Exception as e:
        print(f"Verification failed: {e}")
