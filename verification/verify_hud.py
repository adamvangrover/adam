from playwright.sync_api import sync_playwright
import time
import sys

def run():
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()

        # We need a local server
        url = "http://localhost:8000/showcase/index.html"
        print(f"Navigating to {url}...")
        try:
            page.goto(url)
        except Exception as e:
            print(f"Failed to load page: {e}")
            sys.exit(1)

        # Check for HUD
        print("Checking for ADAM HUD...")
        hud = page.query_selector("#adam-hud")
        if hud:
            print("HUD found!")
            print(f"HUD Text: {hud.inner_text()}")
        else:
            print("HUD NOT found!")
            sys.exit(1)

        # Check for Sector Swarm link
        print("Checking for Sector Swarm link...")
        link = page.query_selector("a[href='sector_swarm.html']")
        if link:
            print("Sector Swarm link found.")
        else:
            print("Sector Swarm link missing!")
            sys.exit(1)

        browser.close()

if __name__ == "__main__":
    run()
