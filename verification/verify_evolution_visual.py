import os
from playwright.sync_api import sync_playwright

def verify_evolution_visual():
    print("Capturing Evolution Hub Screenshot...")

    evo_path = os.path.abspath("showcase/evolution.html")

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(f"file://{evo_path}")

        # Wait for charts
        page.wait_for_selector("#complexityChart", timeout=5000)

        # Take full page screenshot
        page.screenshot(path="verification/evolution_hub.png", full_page=True)
        print("Saved verification/evolution_hub.png")

        # Test Command Palette visual
        page.keyboard.press("Control+k")
        # Wait for animation/transition if any
        page.wait_for_timeout(500)

        page.screenshot(path="verification/evolution_hub_cmd_palette.png")
        print("Saved verification/evolution_hub_cmd_palette.png")

        browser.close()

if __name__ == "__main__":
    verify_evolution_visual()
