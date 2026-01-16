from playwright.sync_api import sync_playwright
import time
import sys
import os

def verify_showcase_updates():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        # Ensure the server is running (assumed port 8000)
        base_url = "http://localhost:8000/showcase"

        # --- TEST 1: Prompt Library Page ---
        print("\n--- Testing Prompt Library (prompts.html) ---")
        try:
            page.goto(f"{base_url}/prompts.html")
            page.wait_for_selector("#prompt-list button", timeout=5000)
            print("✅ Prompt Library loaded.")

            # Count items
            count_text = page.locator("#prompt-count").inner_text()
            print(f"✅ Found {count_text} prompts.")

            # Click first item
            first_prompt = page.locator("#prompt-list button").first
            prompt_name = first_prompt.inner_text().split("\n")[0] # handle span structure
            print(f"Clicking prompt: {prompt_name}")
            first_prompt.click()

            # Verify content viewer
            page.wait_for_selector("#prompt-content pre")
            content = page.locator("#prompt-content").inner_text()
            if len(content) > 10:
                print("✅ Content viewer populated.")
            else:
                print("❌ Content viewer empty!")

            page.screenshot(path="verification/showcase_prompts.png")

        except Exception as e:
            print(f"❌ Failed to verify prompts.html: {e}")

        # --- TEST 2: Financial Twin Portability ---
        print("\n--- Testing Financial Twin Portability ---")
        try:
            page.goto(f"{base_url}/financial_twin.html")
            # Check for Copy Context button
            # Note: The button might be hidden or styled specifically
            copy_btn = page.locator("button:has-text('Copy Context')")
            if copy_btn.is_visible():
                print("✅ 'Copy Context' button visible.")
                copy_btn.click()
                # We can't easily verify clipboard in headless, but if click works without error, it's good.
            else:
                print("❌ 'Copy Context' button NOT found.")

        except Exception as e:
            print(f"❌ Failed to verify financial_twin.html: {e}")

        # --- TEST 3: Institutional Radar Portability ---
        print("\n--- Testing Institutional Radar Portability ---")
        try:
            page.goto(f"{base_url}/institutional_radar.html")
            copy_btn = page.locator("button:has-text('COPY THESIS')")
            if copy_btn.is_visible():
                print("✅ 'COPY THESIS' button visible.")
                copy_btn.click()
            else:
                print("❌ 'COPY THESIS' button NOT found.")

        except Exception as e:
            print(f"❌ Failed to verify institutional_radar.html: {e}")

        browser.close()

if __name__ == "__main__":
    verify_showcase_updates()
