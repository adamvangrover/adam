import os
from playwright.sync_api import sync_playwright

def verify_neural_link():
    cwd = os.getcwd()
    index_path = f"file://{cwd}/showcase/index.html"

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        print(f"Navigating to {index_path}")
        page.goto(index_path)

        # Check for the Neural Link card
        # The link text is "OPEN LINK →" inside a card with title "NEURAL LINK"
        # I can look for the link with href "adam_convergence_live_neural_link.html"

        link = page.locator('a[href="adam_convergence_live_neural_link.html"]')
        if link.count() == 0:
            print("Error: Link to neural link page not found in index.html")
            browser.close()
            return

        print("Found link. Clicking...")
        link.click()

        # Wait for navigation
        page.wait_for_load_state('networkidle')

        print(f"Current URL: {page.url}")

        # Check title
        title = page.title()
        print(f"Page title: {title}")

        if "ADAM-CONVERGENCE" in title:
            print("Verification successful: Title matches.")
        else:
            print(f"Verification warning: Title '{title}' does not match expected 'ADAM-CONVERGENCE'")

        # Take screenshot
        screenshot_path = "verification/neural_link_screenshot.png"
        page.screenshot(path=screenshot_path, full_page=True)
        print(f"Screenshot saved to {screenshot_path}")

        browser.close()

if __name__ == "__main__":
    verify_neural_link()
