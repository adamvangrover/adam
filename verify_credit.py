from playwright.sync_api import sync_playwright

def verify():
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto("http://localhost:8000/comprehensive_credit_dashboard.html")
        page.wait_for_timeout(3000)
        page.screenshot(path="verification_credit.png", full_page=True)
        print("Screenshot saved to verification_credit.png")
        browser.close()

if __name__ == "__main__":
    verify()