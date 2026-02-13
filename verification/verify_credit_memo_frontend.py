import http.server
import socketserver
import threading
import time
import os
import sys
from playwright.sync_api import sync_playwright

PORT = 8081
Handler = http.server.SimpleHTTPRequestHandler

def start_server():
    # Only bind if port is free (retry logic or assume it's free/reuse addr)
    # SimpleHTTPRequestHandler serves from CWD
    try:
        with socketserver.TCPServer(("", PORT), Handler) as httpd:
            print("Serving at port", PORT)
            httpd.serve_forever()
    except OSError:
        print(f"Port {PORT} already in use, assuming server running.")

def verify_frontend():
    # Start server in thread
    server_thread = threading.Thread(target=start_server, daemon=True)
    server_thread.start()

    # Wait for server to start
    time.sleep(2)

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(repo_root) # Ensure relative paths work for server

    url = f"http://localhost:{PORT}/showcase/credit_memo_automation.html"
    print(f"Verifying: {url}")

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.set_viewport_size({"width": 1280, "height": 800})
            page.goto(url)

            # 1. Verify Header
            print("Checking Title...")
            title = page.title()
            print(f"Page Title: {title}")
            assert "Enterprise Credit Memo Automation" in title

            # 2. Verify Memo Loaded (Wait for JS fetch)
            print("Waiting for Memo Content...")
            page.wait_for_selector("#memo-container h1", timeout=10000)
            memo_title = page.inner_text("#memo-container h1")
            print(f"Memo H1: {memo_title}")
            assert "TechCorp Inc." in memo_title

            # 3. Verify Citation Pin
            print("Checking Citation Pin...")
            page.wait_for_selector(".citation-pin", timeout=5000)

            # Take screenshot before click
            screenshot_dir = os.path.join(repo_root, "verification", "screenshots")
            os.makedirs(screenshot_dir, exist_ok=True)
            page.screenshot(path=os.path.join(screenshot_dir, "credit_memo_initial.png"))

            page.click(".citation-pin")

            # 4. Verify PDF Viewer Interaction
            print("Checking PDF Viewer...")
            page.wait_for_selector("#mock-pdf-page", state="visible", timeout=5000)
            doc_text = page.inner_text("#doc-title")
            print(f"Doc Title: {doc_text}")
            assert doc_text != "Waiting for selection..."

            # 5. Verify Audit Log
            print("Checking Audit Log...")
            page.wait_for_selector("#audit-table-body tr", timeout=5000)
            audit_action = page.inner_text("#audit-table-body tr td:nth-child(2)")
            print(f"Latest Audit Action: {audit_action}")
            assert "GENERATE_CREDIT_MEMO" in audit_action

            # Take final screenshot
            final_screenshot = os.path.join(screenshot_dir, "credit_memo_final.png")
            page.screenshot(path=final_screenshot)
            print(f"Screenshot saved to {final_screenshot}")

            print("VERIFICATION SUCCESS: Frontend loaded correctly.")
            browser.close()

    except Exception as e:
        print(f"VERIFICATION FAILED: {e}")
        sys.exit(1)

if __name__ == "__main__":
    verify_frontend()
