import os
import subprocess
import time
from playwright.sync_api import sync_playwright

def verify_system_brain():
    server = subprocess.Popen(["python3", "-m", "http.server", "8000"], cwd="showcase")
    time.sleep(2)

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()

            # Capture console logs
            page.on("console", lambda msg: print(f"Browser Console: {msg.text}"))

            print("Navigating to system_brain.html...")
            page.goto("http://localhost:8000/system_brain.html")

            time.sleep(2)
            os.makedirs("verification_artifacts", exist_ok=True)
            page.screenshot(path="verification_artifacts/debug_load.png")
            print("Captured debug_load.png")

            try:
                # Force fill to bypass strict visibility checks if overlay is issue
                print("Searching for 'generate'...")
                page.fill("#search-input", "generate", force=True)
                page.keyboard.press("Enter") # Search is input event, but just in case
            except Exception as e:
                print(f"Fill failed: {e}")

            time.sleep(2)
            page.screenshot(path="verification_artifacts/debug_after_search.png")

            # Check if inspector opened
            if page.locator("#inspector").get_attribute("class").find("translate-x-full") == -1:
                 print("Inspector is OPEN (translate-x-full removed).")
            else:
                 print("Inspector is CLOSED.")

            # Check content
            content = page.text_content("#insp-content")
            print(f"Inspector Content Preview: {content[:100]}")

    finally:
        server.terminate()

if __name__ == "__main__":
    verify_system_brain()
