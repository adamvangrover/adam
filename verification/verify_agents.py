
import os

from playwright.sync_api import sync_playwright


def run():
    # Start a simple HTTP server to serve the static files
    import http.server
    import socketserver
    import threading
    import time

    PORT = 8000
    Handler = http.server.SimpleHTTPRequestHandler

    # Change directory to showcase/ to serve files correctly relative to root
    # Actually, if I serve from root, then paths like "js/nav.js" in agents.html might break
    # if agents.html is in showcase/ and references "js/nav.js" (which implies showcase/js/nav.js).
    # Let checking paths.
    # agents.html is in showcase/.
    # It references "js/mock_data.js" which means "showcase/js/mock_data.js".
    # So serving from "showcase/" is correct.

    os.chdir("showcase")

    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print("serving at port", PORT)
        server_thread = threading.Thread(target=httpd.serve_forever)
        server_thread.daemon = True
        server_thread.start()

        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                page = browser.new_page()
                page.set_viewport_size({"width": 1280, "height": 800})

                # Navigate to the agents page
                page.goto(f"http://localhost:{PORT}/agents.html")

                # Wait for agents to load (grid)
                page.wait_for_selector(".agent-card")

                # Take screenshot of Grid View
                page.screenshot(path="../verification/agents_grid.png")
                print("Screenshot of Grid View taken.")

                # Click Network View
                page.click("#btn-view-network")
                # Wait for network container to be visible/active
                # (vis-network canvas usually takes a moment)
                time.sleep(2)
                page.screenshot(path="../verification/agents_network.png")
                print("Screenshot of Network View taken.")

                # Open Terminal
                page.click("button[title=\"Toggle Console\"]")
                time.sleep(1)
                page.screenshot(path="../verification/agents_terminal.png")
                print("Screenshot of Terminal taken.")

                # Open Settings
                page.click("button[title=\"Runtime Configuration\"]")
                time.sleep(1)
                page.screenshot(path="../verification/agents_settings.png")
                print("Screenshot of Settings taken.")

        finally:
            httpd.shutdown()

if __name__ == "__main__":
    run()
