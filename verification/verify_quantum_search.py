import os
from playwright.sync_api import sync_playwright

def verify_quantum_search():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        # 1. Verify Dashboard
        # Construct absolute path
        cwd = os.getcwd()
        dashboard_url = f"file://{cwd}/showcase/quantum_search.html"
        print(f"Loading {dashboard_url}...")

        page.goto(dashboard_url)

        # Wait for some content
        page.wait_for_selector("h1:has-text('QUANTUM SEARCH RUNTIME')")
        page.wait_for_selector("#schedule-canvas")

        # Wait a bit for animation to start
        page.wait_for_timeout(2000)

        print("Taking dashboard screenshot...")
        page.screenshot(path="verification/quantum_search_dashboard.png")

        # 2. Verify Navigation Link
        # Open the sidebar if it's not visible (mobile) - but desktop default is hidden until toggle or width?
        # showcasing nav.js logic: "body.has-global-nav { padding-left: 16rem; }" on desktop
        # The nav is transformed -translate-x-full by default on small screens?
        # Let's set viewport to desktop
        page.set_viewport_size({"width": 1280, "height": 720})

        # Wait for nav to be injected
        page.wait_for_selector("#side-nav")

        # Check for the new link
        link = page.get_by_role("link", name="Quantum Search")
        if link.is_visible():
            print("Quantum Search link found!")
            # Highlight it
            link.scroll_into_view_if_needed()
            page.screenshot(path="verification/quantum_search_nav.png")
        else:
            print("Quantum Search link NOT visible/found.")

        browser.close()

if __name__ == "__main__":
    if not os.path.exists("verification"):
        os.makedirs("verification")
    verify_quantum_search()
