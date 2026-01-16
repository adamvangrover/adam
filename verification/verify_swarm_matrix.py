from playwright.sync_api import sync_playwright
import sys

def run():
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()

        url = "http://localhost:8000/showcase/sector_swarm.html"
        print(f"Navigating to {url}...")
        page.goto(url)

        # Wait for JS to load
        page.wait_for_selector("#matrix-body tr")

        # Check matrix rows
        rows = page.query_selector_all("#matrix-body tr")
        print(f"Found {len(rows)} matrix rows.")

        if len(rows) != 3:
            print("Expected 3 rows in impact matrix!")
            sys.exit(1)

        # Check for heat map coloring
        cells = page.query_selector_all("#matrix-body td")
        print(f"Found {len(cells)} matrix cells.")

        # Take screenshot
        page.screenshot(path="verification_artifacts/swarm_matrix.png")
        print("Screenshot saved to verification_artifacts/swarm_matrix.png")

        browser.close()

if __name__ == "__main__":
    run()
