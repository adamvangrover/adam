from playwright.sync_api import sync_playwright

def verify_showcase():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        # 1. Verify Index Page
        print("Navigating to showcase/index.html...")
        page.goto("http://localhost:8000/showcase/index.html")
        page.wait_for_selector(".cyber-card") # Wait for cards to load

        print("Taking index screenshot...")
        page.screenshot(path="verification/showcase_index.png", full_page=True)

        # 2. Verify Trading Page
        print("Navigating to showcase/trading.html...")
        page.goto("http://localhost:8000/showcase/trading.html")
        page.wait_for_selector("#chart-container canvas") # Wait for chart

        # Wait a bit for simulation to trigger updates
        page.wait_for_timeout(2000)

        print("Taking trading screenshot...")
        page.screenshot(path="verification/showcase_trading.png", full_page=True)

        browser.close()

if __name__ == "__main__":
    verify_showcase()
