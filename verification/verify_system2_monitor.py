import os
from playwright.sync_api import sync_playwright

def verify_system2_monitor():
    file_path = os.path.abspath("showcase/high_conviction_monitor.html")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(f"file://{file_path}")

        # Check Title
        title = page.title()
        print(f"Page Title: {title}")
        assert "HIGH CONVICTION MONITOR" in title

        # Check if Grid is populated (implies JS ran)
        page.wait_for_selector(".asset-card")
        cards = page.query_selector_all(".asset-card")
        print(f"Asset Cards found: {len(cards)}")
        assert len(cards) > 0

        # Check for Live Indicator
        indicator = page.query_selector(".live-indicator")
        assert indicator is not None, "Live indicator not found"

        # Check for Timestamp
        timestamp = page.inner_text("#lastUpdated")
        print(f"Timestamp: {timestamp}")
        assert "LAST UPDATE" in timestamp or "CONNECTING" in timestamp

        print("System 2 Monitor verification passed.")
        browser.close()

if __name__ == "__main__":
    verify_system2_monitor()
