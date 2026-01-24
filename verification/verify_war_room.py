from playwright.sync_api import sync_playwright
import time
import os

OUTPUT_DIR = "verification"
SCREENSHOT_PATH = os.path.join(OUTPUT_DIR, "war_room_verification.png")

def verify_war_room():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={'width': 1280, 'height': 800})

        # Navigate to War Room
        print("Navigating to War Room...")
        page.goto("http://localhost:8080/showcase/war_room.html")

        # Verify Header
        print("Verifying Header...")
        page.wait_for_selector("h1:has-text('STRATEGIC WAR ROOM')")

        # Select Scenario
        print("Selecting Scenario...")
        page.click("#card_sim_semi_blockade")

        # Verify Scenario Selected Status
        page.wait_for_selector("#simulation-status:has-text('SCENARIO SELECTED')")

        # Initiate Simulation
        print("Initiating Simulation...")
        page.click("button:has-text('INITIATE SIMULATION')")

        # Wait for some events
        print("Waiting for events...")
        time.sleep(3)

        # Verify events appear
        page.wait_for_selector(".event-item")

        # Take Screenshot
        print(f"Taking screenshot at {SCREENSHOT_PATH}...")
        page.screenshot(path=SCREENSHOT_PATH)

        browser.close()
        print("Verification Complete.")

if __name__ == "__main__":
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    verify_war_room()
