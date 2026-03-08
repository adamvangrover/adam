import time
from playwright.sync_api import sync_playwright

def verify():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1280, "height": 720})
        # Wait a moment just in case
        time.sleep(1)
        page.goto("http://localhost:8002/showcase/apps/market_data.html")
        page.wait_for_load_state("networkidle")

        # Ingest data
        page.click("button:has-text('Load Sample')")
        page.click("button:has-text('INGEST')")
        time.sleep(1)

        page.screenshot(path="market_data_before.png")
        print("Screenshot before scrub saved to market_data_before.png")

        # Run scrub
        page.click("button:has-text('ML SCRUB')")
        time.sleep(2.5) # Wait for animation

        page.screenshot(path="market_data_after.png")
        print("Screenshot after scrub saved to market_data_after.png")

        browser.close()

verify()
