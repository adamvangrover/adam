from playwright.sync_api import sync_playwright, expect
import os

def verify_nexus():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        # Navigate to Nexus page
        page.goto("http://localhost:8080/showcase/nexus.html")

        # Verify title
        expect(page).to_have_title("ADAM | NEXUS Simulation")

        # Wait for simulation data to load (graph nodes should appear)
        # We can look for the canvas element of vis-network
        page.wait_for_selector("#nexus-network canvas")

        # Check for the sidebar stream
        expect(page.get_by_text("SYNTHETIC COGNITIVE STREAM")).to_be_visible()

        # Check that the index link exists in main page
        page.goto("http://localhost:8080/showcase/index.html")
        expect(page.get_by_role("link", name="ENTER NEXUS")).to_be_visible()

        # Go back to Nexus and screenshot
        page.goto("http://localhost:8080/showcase/nexus.html")
        page.wait_for_timeout(2000) # Wait for animation/layout

        screenshot_path = "verification/nexus_verification.png"
        page.screenshot(path=screenshot_path)
        print(f"Screenshot saved to {screenshot_path}")

        browser.close()

if __name__ == "__main__":
    verify_nexus()
