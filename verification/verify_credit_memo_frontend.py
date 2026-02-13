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
    try:
        with socketserver.TCPServer(("", PORT), Handler) as httpd:
            print("Serving at port", PORT)
            httpd.serve_forever()
    except OSError:
        print(f"Port {PORT} already in use, assuming server running.")

def verify_frontend():
    server_thread = threading.Thread(target=start_server, daemon=True)
    server_thread.start()
    time.sleep(2)

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(repo_root)

    url = f"http://localhost:{PORT}/showcase/credit_memo_automation.html"
    print(f"Verifying: {url}")

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.set_viewport_size({"width": 1280, "height": 800})
            page.goto(url)

            print("Checking Title...")
            title = page.title()
            assert "Enterprise Credit Memo Automation" in title

            print("Waiting for Memo Content...")
            page.wait_for_selector("#memo-container h1", timeout=15000)
            memo_title = page.inner_text("#memo-container h1")
            print(f"Memo H1: {memo_title}")
            assert any(x in memo_title for x in ["TechCorp", "Apple", "Tesla", "JPMorgan"]), f"Unexpected memo title: {memo_title}"

            # Verify Agent Badges
            print("Checking Agent Badges...")
            page.wait_for_selector(".fa-robot", timeout=5000)
            badge_text = page.inner_text(".text-purple-700") # Writer Badge
            print(f"Agent Badge: {badge_text}")
            assert "WRITER" in badge_text

            # Verify Tabs
            print("Checking Tabs...")
            page.click("#btn-tab-annex-a")
            time.sleep(0.5)

            # Verify Annex A (Historicals)
            print("Checking Annex A (Financials)...")
            table_visible = page.is_visible("#financials-table")
            assert table_visible, "Financials table not visible"
            table_text = page.inner_text("#financials-table")
            assert "Revenue" in table_text
            assert "FY2025" in table_text or "FY2024" in table_text

            page.click("#btn-tab-annex-b")
            time.sleep(0.5)

            # Verify Annex B (DCF)
            print("Checking Annex B (Valuation)...")
            dcf_visible = page.is_visible("#dcf-container")
            assert dcf_visible, "DCF container not visible"
            dcf_text = page.inner_text("#dcf-container")
            assert "WACC" in dcf_text
            assert "IMPLIED SHARE PRICE" in dcf_text or "Implied Share Price" in dcf_text
            assert "$" in dcf_text

            print("VERIFICATION SUCCESS: Frontend Tabs, Historicals, DCF, and Agent Badges verified.")
            browser.close()

    except Exception as e:
        print(f"VERIFICATION FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    verify_frontend()
