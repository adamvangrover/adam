import os
from playwright.sync_api import sync_playwright

def verify_nav_xss():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        # Load showcase/data.html
        # We need absolute path
        cwd = os.getcwd()
        file_url = f"file://{cwd}/showcase/data.html"
        print(f"Loading {file_url}")
        page.goto(file_url)

        # Inject malicious entry into REPO_DATA
        print("Injecting malicious data...")
        page.evaluate("""
            if (!window.REPO_DATA) window.REPO_DATA = { nodes: [] };
            window.REPO_DATA.nodes.push({
                id: 99999,
                label: '<img src=x onerror=alert("XSS")>',
                group: 'js',
                path: 'malicious.js',
                value: 5
            });
        """)

        # Listen for dialogs (alerts)
        page.on("dialog", lambda dialog: print(f"ALERT DETECTED: {dialog.message}"))

        # Find search input
        # It's in the nav injected by nav.js. id="global-search"
        # Wait for nav to load
        print("Waiting for nav to load...")
        page.wait_for_selector("#global-search")

        # Type search query
        print("Searching for malicious entry...")
        page.fill("#global-search", "<img")

        # Wait for results
        page.wait_for_selector("#search-results a")

        # Get result text
        result_text = page.inner_text("#search-results")
        print(f"Result text: {result_text}")

        # Take screenshot
        screenshot_path = f"{cwd}/verification/nav_xss_verification.png"
        page.screenshot(path=screenshot_path)
        print(f"Screenshot saved to {screenshot_path}")

        # Check if the HTML tag is rendered as text
        # If it's rendered as text, inner_text should contain the tag brackets
        # If it's HTML, inner_text might not show them or show broken image

        # But specifically, we want to ensure NO ALERT happened (handled by event listener printing)
        # And we can check the HTML content of the result container

        result_html = page.inner_html("#search-results")
        print(f"Result HTML: {result_html}")

        if "&lt;img" in result_html or "&lt;img" in result_html:
             print("SUCCESS: HTML tags are escaped.")
        elif "<img" in result_html:
             # Check if it is inside a text node or actual tag
             # inner_html returns the HTML string. If we see <img src=...> it means it IS a tag.
             # Wait, if we use textContent, the innerHTML of the parent will show encoded entities?
             # No. If we use textContent = "<img...", the browser converts < to &lt; in the DOM.
             # So innerHTML will return "&lt;img...".
             # So if we see "&lt;" we are good.
             pass
        else:
             print("WARNING: Malicious tag not found or something else happened.")

        browser.close()

if __name__ == "__main__":
    verify_nav_xss()
