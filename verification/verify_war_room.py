from playwright.sync_api import sync_playwright
import time

def verify_war_room():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={'width': 1920, 'height': 1080})

        # Navigate to the War Room page
        page.goto("http://localhost:8000/showcase/war_room_v2.html")

        # Wait for the ticker to load (it's populated by JS)
        page.wait_for_selector("#ticker-feed .ticker-row")

        # Wait for the news feed
        page.wait_for_selector("#news-feed .news-item")

        # Wait a bit for the canvas animation or just take a screenshot
        time.sleep(2)

        # Screenshot
        page.screenshot(path="verification/war_room_v2.png")
        browser.close()

if __name__ == "__main__":
    verify_war_room()
