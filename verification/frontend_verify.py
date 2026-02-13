import os
import sys
import time
import subprocess
from playwright.sync_api import sync_playwright

def verify_frontend():
    print("Starting Frontend Verification...")
    os.makedirs("verification_screenshots", exist_ok=True)

    # Start Server
    port = 8092
    server_process = subprocess.Popen(
        [sys.executable, "-m", "http.server", str(port)],
        cwd=os.path.abspath("showcase"),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    time.sleep(2)

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(viewport={'width': 1920, 'height': 1080})
            page = context.new_page()

            # Verify credit_memo.html
            print("Verifying credit_memo.html")
            page.goto(f"http://localhost:{port}/credit_memo.html")

            # Select Apple
            try:
                page.select_option("#borrower-select", label="Apple Inc. (Risk: 75)")
            except:
                page.locator("#borrower-select").select_option(index=1)

            page.click("#generate-btn")

            # Wait for generation
            try:
                page.wait_for_function("document.body.innerText.includes('Analysis Complete')", timeout=20000)
            except:
                pass

            # Screenshot
            page.screenshot(path="verification_screenshots/credit_memo_final.png")
            print("Screenshot saved: verification_screenshots/credit_memo_final.png")

            # Verify credit_memo_v2.html
            print("Verifying credit_memo_v2.html")
            page.goto(f"http://localhost:{port}/credit_memo_v2.html")

            # Select Apple
            try:
                page.select_option("#borrower-select", label="Apple Inc. (Risk: 75)")
            except:
                page.locator("#borrower-select").select_option(index=1)

            page.click("#generate-btn")

             # Wait for generation
            try:
                page.wait_for_function("document.body.innerText.includes('Analysis Complete')", timeout=20000)
            except:
                pass

            # Screenshot
            page.screenshot(path="verification_screenshots/credit_memo_v2_final.png")
            print("Screenshot saved: verification_screenshots/credit_memo_v2_final.png")

    finally:
        server_process.kill()
        print("Server stopped.")

if __name__ == "__main__":
    verify_frontend()
