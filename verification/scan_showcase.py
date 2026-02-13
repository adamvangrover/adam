import http.server
import socketserver
import threading
import time
import os
import sys
from playwright.sync_api import sync_playwright

PORT = 8085
Handler = http.server.SimpleHTTPRequestHandler

def start_server():
    try:
        # Use a socket server to retry binding if necessary or assume it's free
        with socketserver.TCPServer(("", PORT), Handler) as httpd:
            print(f"Serving at port {PORT}")
            httpd.serve_forever()
    except OSError:
        print(f"Port {PORT} already in use, reusing.")

def scan_showcase():
    # Start server in thread
    server_thread = threading.Thread(target=start_server, daemon=True)
    server_thread.start()
    time.sleep(2) # Wait for server

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(repo_root)

    showcase_dir = os.path.join(repo_root, "showcase")
    files = [f for f in os.listdir(showcase_dir) if f.endswith(".html")]

    report = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        for filename in files:
            url = f"http://localhost:{PORT}/showcase/{filename}"
            print(f"Checking {filename}...")

            errors = []

            # Listen for console errors
            def handle_console(msg):
                if msg.type == "error":
                    errors.append(f"CONSOLE: {msg.text}")
            page.on("console", handle_console)

            # Listen for network failures
            def handle_response(response):
                if not response.ok: # 404, 500, etc.
                    # Ignore known benign issues (e.g. favicon)
                    if "favicon.ico" in response.url: return
                    errors.append(f"NETWORK {response.status}: {response.url}")
            page.on("response", handle_response)

            try:
                page.goto(url, timeout=3000, wait_until="domcontentloaded")
                # Wait a bit for JS to execute
                page.wait_for_timeout(500)

                title = page.title()
                if not title:
                    errors.append("WARNING: No Page Title")

            except Exception as e:
                errors.append(f"EXCEPTION: {str(e)}")

            # Clean up listeners
            page.remove_listener("console", handle_console)
            page.remove_listener("response", handle_response)

            if errors:
                report.append({"file": filename, "errors": errors})
            else:
                print(f"PASS: {filename}")

        browser.close()

    print("\n=== SHOWCASE SCAN REPORT ===")
    if not report:
        print("All files passed basic checks!")
    else:
        for item in report:
            print(f"\nFILE: {item['file']}")
            for err in item['errors']:
                print(f"  - {err}")

if __name__ == "__main__":
    scan_showcase()
