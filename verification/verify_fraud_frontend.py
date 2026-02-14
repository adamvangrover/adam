import time
import subprocess
import os
from playwright.sync_api import sync_playwright

def verify_fraud_dashboard():
    # Start HTTP server
    # We run from root, so showcase is at /showcase/
    server = subprocess.Popen(["python3", "-m", "http.server", "8085"], cwd=".", stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(3) # Wait for server

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()

            print("Verifying Fraud Dashboard...")
            # 1. Fraud Investigation
            page.goto("http://localhost:8085/showcase/fraud_investigation.html")
            page.wait_for_selector("#companyName", timeout=5000)

            # Check title
            title = page.title()
            print(f"Title: {title}")
            if "FORENSIC LAB" not in title:
                print("ERROR: Title mismatch")

            # Check if cases loaded (Sidebar should have items)
            try:
                page.wait_for_selector(".case-file", timeout=5000)
                cases = page.query_selector_all(".case-file")
                print(f"Found {len(cases)} cases.")
            except Exception as e:
                print(f"Error finding cases: {e}")

            # Take screenshot of Dashboard
            page.screenshot(path="verification_screenshots/fraud_dashboard.png")
            print("Screenshot saved: verification_screenshots/fraud_dashboard.png")

            print("Verifying Market Mayhem Links...")
            # 2. Market Mayhem Rebuild (Check links)
            page.goto("http://localhost:8085/showcase/market_mayhem_rebuild.html")

            # Check for "FORENSIC LAB" button
            # We use text content match
            content = page.content()
            if "FORENSIC LAB" in content:
                print("Found 'FORENSIC LAB' link.")
            else:
                print("ERROR: 'FORENSIC LAB' link not found.")

            if "TERMINAL" in content:
                print("Found 'TERMINAL' link.")
            else:
                print("ERROR: 'TERMINAL' link not found.")

            page.screenshot(path="verification_screenshots/market_mayhem_links.png")
            print("Screenshot saved: verification_screenshots/market_mayhem_links.png")

            print("Verifying System Terminal...")
            # 3. Terminal
            page.goto("http://localhost:8085/showcase/system_terminal.html")
            page.wait_for_selector("#cmdInput")
            page.fill("#cmdInput", "/seed Testing Injection")
            page.press("#cmdInput", "Enter")
            time.sleep(2) # Wait for simulation

            page.screenshot(path="verification_screenshots/terminal_interaction.png")
            print("Screenshot saved: verification_screenshots/terminal_interaction.png")

    except Exception as e:
        print(f"Verification Failed: {e}")
    finally:
        server.terminate()

if __name__ == "__main__":
    os.makedirs("verification_screenshots", exist_ok=True)
    verify_fraud_dashboard()
