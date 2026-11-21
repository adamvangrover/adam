import time
from playwright.sync_api import sync_playwright

def verify():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1280, "height": 720})
        # Wait a moment just in case
        time.sleep(1)
        page.goto("http://localhost:8002/showcase/apps/portfolio_attribution.html")
        page.wait_for_load_state("networkidle")
        time.sleep(2)
        page.screenshot(path="portfolio_attr.png")
        print("Screenshot saved to portfolio_attr.png")
        browser.close()

verify()
