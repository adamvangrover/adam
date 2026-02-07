import os
from playwright.sync_api import sync_playwright

def verify_narrative_dashboard():
    file_path = os.path.abspath("showcase/narrative_dashboard_preview.html")
    url = f"file://{file_path}"
    print(f"Navigating to: {url}")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.set_viewport_size({"width": 1000, "height": 600})
        page.goto(url)

        # Verify cards exist
        cards = page.locator(".card")
        count = cards.count()
        print(f"Found {count} narrative cards.")

        if count >= 3:
            print("Verification Successful: Narrative cards rendered.")
        else:
            print("Verification Failed: Cards missing.")

        output_dir = os.path.abspath("verification")
        os.makedirs(output_dir, exist_ok=True)
        page.screenshot(path=os.path.join(output_dir, "narrative_dashboard.png"))
        browser.close()

if __name__ == "__main__":
    verify_narrative_dashboard()
