import os
from playwright.sync_api import sync_playwright

def verify_evolution():
    print("Verifying Evolution Hub...")

    evo_path = os.path.abspath("showcase/evolution.html")
    if not os.path.exists(evo_path):
        print(f"FAILED: {evo_path} does not exist.")
        return

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(f"file://{evo_path}")

        # 1. Check Header
        if page.is_visible("h1:has-text('ADAM EVOLUTION HUB')"):
            print("PASSED: Header found.")
        else:
            print("FAILED: Header mismatch.")

        # 2. Check Timeline
        if page.is_visible(".timeline"):
            print("PASSED: Timeline container found.")
        else:
            print("FAILED: Timeline missing.")

        # 3. Check Charts
        if page.is_visible("#complexityChart") and page.is_visible("#cognitiveChart"):
            print("PASSED: Performance charts found.")
        else:
            print("FAILED: Charts missing.")

        # 4. Check Constraints
        if page.is_visible(".constraints-grid"):
            print("PASSED: Constraints grid found.")
        else:
            print("FAILED: Constraints grid missing.")

        # 5. Check Command Palette Trigger (Nav)
        # Note: Nav loads async via script, might need wait
        page.wait_for_selector("#side-nav", timeout=5000)

        # Simulate Ctrl+K
        print("Simulating Ctrl+K...")
        page.keyboard.press("Control+k")

        # Check if overlay appears
        # It has class 'hidden' initially, removed on toggle
        try:
            page.wait_for_selector("#command-palette-overlay.active", timeout=2000)
            print("PASSED: Command Palette overlay activated via Ctrl+K.")
        except:
            # Check if it removed 'hidden' class at least
            palette = page.query_selector("#command-palette-overlay")
            classes = palette.get_attribute("class")
            if "hidden" not in classes:
                 print("PASSED: Command Palette overlay became visible (hidden class removed).")
            else:
                 print(f"FAILED: Command Palette did not activate. Classes: {classes}")

        browser.close()

if __name__ == "__main__":
    verify_evolution()
