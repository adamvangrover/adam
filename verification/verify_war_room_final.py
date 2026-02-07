import os
from playwright.sync_api import sync_playwright

def verify_war_room():
    file_path = os.path.abspath("showcase/war_room_v2.html")
    url = f"file://{file_path}"
    print(f"Navigating to: {url}")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.set_viewport_size({"width": 1920, "height": 1080})
        page.goto(url)
        try:
            page.wait_for_selector("#ticker-feed .ticker-row", timeout=5000)
            print("Ticker populated.")
        except:
            print("Ticker did not populate (timeout).")

        output_dir = os.path.abspath("verification")
        os.makedirs(output_dir, exist_ok=True)
        page.screenshot(path=os.path.join(output_dir, "war_room_final.png"))
        browser.close()

if __name__ == "__main__":
    verify_war_room()
