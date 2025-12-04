from playwright.sync_api import sync_playwright
import time

def verify_deep_dive_simulation():
    with sync_playwright() as p:
        # Launch browser
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={'width': 1280, 'height': 800})
        page = context.new_page()

        print("Navigating to deep_dive.html...")
        page.goto("http://localhost:8085/deep_dive.html")
        time.sleep(2)  # Wait for initial load

        # 1. Take initial screenshot
        print("Taking initial screenshot...")
        page.screenshot(path="verification/deep_dive_initial.png")

        # 2. Click the 'NEW' simulation button
        print("Clicking 'NEW' button...")
        # Finding the button by text content 'NEW' inside the aside
        page.get_by_role("button", name="NEW").click()

        # 3. Wait for modal to appear and take screenshot
        print("Waiting for modal...")
        # Wait for modal to be visible (class hidden removed)
        page.wait_for_selector("#sim-modal:not(.hidden)")
        time.sleep(1)
        page.screenshot(path="verification/deep_dive_modal.png")

        # 4. Wait for simulation to finish (modal hidden again)
        print("Waiting for simulation to complete...")
        # The simulation runs for about 5-6 seconds. We wait for it to become hidden.
        page.wait_for_selector("#sim-modal", state="hidden", timeout=15000)
        time.sleep(1) # Wait for DOM update

        # 5. Verify new report is in the list
        print("Verifying new report...")
        # Check for "Synthetic Corp" in the sidebar
        expect_loc = page.locator("#report-list")
        if "Synthetic Corp" in expect_loc.inner_text():
            print("SUCCESS: Synthetic Corp found in sidebar.")
        else:
            print("FAILURE: Synthetic Corp NOT found in sidebar.")

        # 6. Take final screenshot
        print("Taking final screenshot...")
        page.screenshot(path="verification/deep_dive_final.png")

        browser.close()

if __name__ == "__main__":
    verify_deep_dive_simulation()
