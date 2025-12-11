from playwright.sync_api import sync_playwright

def verify_apps():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        # Test Analyst OS Dock
        page.goto("http://localhost:8080/showcase/analyst_os.html")
        page.wait_for_selector(".dock-item")
        page.screenshot(path="verification/analyst_os.png")

        # Test Comps
        page.goto("http://localhost:8080/showcase/apps/comps.html")
        page.wait_for_selector("table")
        page.screenshot(path="verification/comps.png")

        # Test Ratios
        page.goto("http://localhost:8080/showcase/apps/ratios.html")
        page.wait_for_selector(".result-box")
        page.screenshot(path="verification/ratios.png")

        # Test Black-Scholes
        page.goto("http://localhost:8080/showcase/apps/black_scholes.html")
        page.wait_for_selector("canvas")
        page.screenshot(path="verification/black_scholes.png")

        # Test Loan
        page.goto("http://localhost:8080/showcase/apps/loan.html")
        page.wait_for_selector("table")
        page.screenshot(path="verification/loan.png")

        # Test M&A
        page.goto("http://localhost:8080/showcase/apps/ma_model.html")
        page.wait_for_selector("canvas")
        page.screenshot(path="verification/ma_model.png")

        browser.close()

if __name__ == "__main__":
    verify_apps()
