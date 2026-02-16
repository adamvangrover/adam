import json
import os
import sys
import subprocess
import time
from playwright.sync_api import sync_playwright

def verify_conviction_paper():
    # 1. Verify JSON Content
    print("Verifying JSON content...")
    try:
        with open("showcase/data/market_mayhem_index.json", "r") as f:
            data = json.load(f)

        # Filter same as JS logic
        valid_articles = [a for a in data if a.get("full_body") and len(a["full_body"]) > 50]
        if not valid_articles:
            print("ERROR: No valid articles found in JSON.")
            sys.exit(1)

        print(f"Found {len(valid_articles)} valid articles.")

    except Exception as e:
        print(f"Error verifying JSON: {e}")
        sys.exit(1)

    # 2. Verify Site Map
    print("\nVerifying Site Map...")
    try:
        with open("showcase/site_map.json", "r") as f:
            sitemap = json.load(f)

        found = False
        for cat in sitemap.get("categories", []):
            for item in cat.get("items", []):
                if item.get("link") == "market_mayhem_conviction.html":
                    found = True
                    break
            if found: break

        if found:
            print("Site Map Verified: 'market_mayhem_conviction.html' found.")
        else:
            print("ERROR: 'market_mayhem_conviction.html' not found in site_map.json.")
            sys.exit(1)

    except Exception as e:
        print(f"Error verifying Site Map: {e}")
        sys.exit(1)

    # 3. Verify UI Elements via Playwright
    print("\nVerifying UI elements...")

    # Start HTTP Server
    server_process = subprocess.Popen(
        [sys.executable, "-m", "http.server", "8000"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        cwd=os.getcwd()
    )
    time.sleep(2) # Wait for server to start

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context()

            # --- TEST 1: Standard Load & UI Structure ---
            print("\n--- TEST 1: Standard Load & UI Structure ---")
            page = context.new_page()
            page.goto("http://localhost:8000/showcase/market_mayhem_conviction.html")

            try:
                page.wait_for_selector(".paper-sheet", timeout=5000)
                print("Content loaded successfully.")
            except Exception as e:
                print(f"Error waiting for content: {e}")
                sys.exit(1)

            # Check Sidebar
            if page.query_selector(".sidebar-toc"):
                print("Sidebar TOC found.")
            else:
                print("ERROR: Sidebar TOC not found.")
                sys.exit(1)

            # Check New Buttons
            if page.query_selector("#terminalBtn"):
                print("Terminal Button found.")
            else:
                print("ERROR: Terminal Button not found.")
                sys.exit(1)

            if page.query_selector("#trendBtn"):
                print("Trends Button found.")
            else:
                print("ERROR: Trends Button not found.")
                sys.exit(1)

            if page.query_selector("#glitchBtn"):
                print("Glitch Button found.")
            else:
                print("ERROR: Glitch Button not found.")
                sys.exit(1)

            # --- TEST 2: Deep Linking ---
            print("\n--- TEST 2: Deep Linking ---")
            # Navigate to specific page (e.g., page 2)
            page.goto("http://localhost:8000/showcase/market_mayhem_conviction.html?page=2")
            page.wait_for_selector(".paper-sheet", timeout=5000)

            # Verify we are on page 2
            indicator = page.inner_text("#pageIndicator")
            print(f"Page Indicator: {indicator}")

            if "PAGE 2" in indicator:
                print("Deep Linking Verified: Successfully loaded Page 2.")
            else:
                print(f"ERROR: Deep Linking failed. Expected Page 2, got {indicator}")

            # --- TEST 3: Interactivity (Redaction) ---
            print("\n--- TEST 3: Interactivity (Redaction) ---")

            # Click Redact Tool
            page.click("#redactTool")

            # Find a paragraph or header to click
            target_selector = ".paper-sheet h1.headline"
            if page.query_selector(target_selector):
                page.click(target_selector)

                # Check if class was added
                is_redacted = page.eval_on_selector(target_selector, "el => el.classList.contains('redacted-text')")
                if is_redacted:
                    print("Redaction Verified: Element correctly redacted on click.")
                else:
                    print("ERROR: Redaction failed. Class 'redacted-text' not applied.")
                    sys.exit(1)
            else:
                print("Warning: Could not find headline to test redaction.")

            # --- TEST 4: Terminal Mode ---
            print("\n--- TEST 4: Terminal Mode ---")
            page.click("#terminalBtn")
            is_terminal = page.eval_on_selector("body", "el => el.classList.contains('terminal-mode')")
            if is_terminal:
                print("Terminal Mode Verified: Body class applied.")
            else:
                print("ERROR: Terminal Mode toggle failed.")
                sys.exit(1)

            # Take screenshot of Terminal Mode
            os.makedirs("verification_screenshots", exist_ok=True)
            screenshot_path = "verification_screenshots/conviction_paper_terminal.png"
            page.screenshot(path=screenshot_path, full_page=True)
            print(f"Terminal Screenshot saved to {screenshot_path}")

            # Toggle back to normal
            page.click("#terminalBtn")

            # --- TEST 5: Trend Chart Modal ---
            print("\n--- TEST 5: Trend Chart Modal ---")
            page.click("#trendBtn")
            is_modal_visible = page.is_visible("#trendModal")
            if is_modal_visible:
                print("Trend Modal Verified: Modal is visible.")
            else:
                print("ERROR: Trend Modal failed to open.")
                sys.exit(1)

            # --- TEST 6: Glitch Effect ---
            print("\n--- TEST 6: Glitch Effect ---")
            # Close modal first using specific selector or JS
            page.eval_on_selector("#trendModal", "el => el.style.display = 'none'")

            page.click("#glitchBtn")
            print("Glitch Button clicked successfully.")

            # Verify Random Glitch Initialization
            print("\n--- TEST 7: Random Glitch Logic ---")
            is_glitch_timer_set = page.evaluate("() => typeof glitchTimer !== 'undefined' && glitchTimer !== null")
            if is_glitch_timer_set:
                print("Random Glitch Logic Verified: glitchTimer is initialized.")
            else:
                print("ERROR: glitchTimer not initialized.")
                sys.exit(1)

            browser.close()

    finally:
        server_process.kill()

if __name__ == "__main__":
    verify_conviction_paper()
