import os
import sys
import time
import subprocess
from playwright.sync_api import sync_playwright

def verify_file(filename, port):
    print(f"--- Verifying {filename} ---")
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()

            url = f"http://localhost:{port}/{filename}"
            print(f"Navigating to {url}...")
            page.goto(url)

            # Check Title
            assert "ADAM" in page.title()
            print("PASS: Page title verified.")

            # Check Tools (Edit Mode / Export)
            print("Checking Tools...")
            # We expect these buttons to exist now in both files
            assert page.is_visible("#edit-btn")
            assert page.is_visible("#export-btn")
            print("PASS: Tools buttons visible.")

            # Check Risk Panel Placeholder
            print("Checking Risk Panel...")
            assert page.is_visible("#risk-quant-panel")
            print("PASS: Risk Panel visible.")

            # Select Apple and Generate
            print("Selecting Apple Inc. and generating...")
            # Use label selection as value might be ID or filename
            # Note: 75.0 becomes 75 in JS string interpolation usually
            try:
                page.select_option("#borrower-select", label="Apple Inc. (Risk: 75)")
            except:
                # Fallback if label is slightly different
                print("Label match failed, trying partial match...")
                page.locator("#borrower-select").select_option(index=1)

            page.click("#generate-btn")

            # Wait for Agent Trace & Memo
            print("Monitoring Agent Terminal...")
            try:
                page.wait_for_function("document.body.innerText.includes('Analysis Complete')", timeout=20000)
            except Exception as e:
                print("Timeout waiting for analysis completion.")
                print("Terminal Content:", page.inner_text("#agent-terminal"))
                raise e
            print("PASS: Agent workflow completed.")

            # Check Risk Panel Content
            # It should now have "RISK METRICS"
            print("Checking Risk Panel Content...")
            risk_content = page.inner_text("#risk-quant-panel")
            assert "RISK METRICS" in risk_content
            assert "PD (1Y)" in risk_content
            print("PASS: Risk Panel populated.")

            print(f"--- {filename} Verified Successfully ---\n")

    except Exception as e:
        print(f"FAIL: Verification failed for {filename}: {e}")
        raise e

def verify_credit_ui_extended():
    print("Starting Extended Credit UI Verification...")

    # Start Server
    port = 8091
    server_process = subprocess.Popen(
        [sys.executable, "-m", "http.server", str(port)],
        cwd=os.path.abspath("showcase"),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    # Wait for server
    time.sleep(2)

    try:
        verify_file("credit_memo.html", port)
        verify_file("credit_memo_v2.html", port)
    finally:
        server_process.kill()
        print("Server stopped.")

if __name__ == "__main__":
    verify_credit_ui_extended()
