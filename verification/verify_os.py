from playwright.sync_api import sync_playwright


def verify_analyst_os():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={'width': 1280, 'height': 800})
        page = context.new_page()

        # Navigate to Analyst OS
        page.goto("http://localhost:8080/showcase/analyst_os.html")

        # Wait for the intro modal and click initialize
        page.get_by_text("INITIALIZE WORKBENCH").click()

        # Wait for the desktop to load
        page.wait_for_timeout(1000)

        # Open DCF app
        page.locator(".dock-item[data-title='DCF Valuator']").click()
        page.wait_for_timeout(1000)

        # Screenshot DCF with new features
        # Note: We need to switch to the iframe to see inside, but for full OS screenshot we can just snap the page
        page.screenshot(path="verification/analyst_os_dcf.png")

        # Close DCF (Click red dot in window bar)
        # We need to find the close button inside the window-bar.
        # Since multiple might exist if we clicked multiple times, let's target the active one.
        # But for simplicity, let's just open WACC now.

        # Open WACC app
        page.locator(".dock-item[data-title='WACC Calculator']").click()
        page.wait_for_timeout(1000)

        # Screenshot WACC
        page.screenshot(path="verification/analyst_os_wacc.png")

        # Open Notepad
        page.locator(".dock-item[data-title='Secure Notepad']").click()
        page.wait_for_timeout(1000)

        # Type something in notepad
        # Finding the notepad window. It has a specific title.
        # We need to access the window content which is NOT an iframe for notepad, it's direct HTML injection.
        page.locator("#secure-note-area").fill("Confidential Analysis: Buy Target A.")
        page.wait_for_timeout(500)

        # Screenshot Notepad
        page.screenshot(path="verification/analyst_os_notepad.png")

        browser.close()

if __name__ == "__main__":
    verify_analyst_os()
