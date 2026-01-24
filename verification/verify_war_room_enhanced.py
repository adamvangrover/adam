from playwright.sync_api import sync_playwright
import time
import os

OUTPUT_DIR = "verification"
SCREENSHOT_PATH = os.path.join(OUTPUT_DIR, "war_room_enhanced_verification.png")

def verify_enhanced_war_room():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={'width': 1280, 'height': 800})

        # Navigate to War Room
        print("Navigating to War Room...")
        page.goto("http://localhost:8080/showcase/war_room.html")

        # Wait for page load
        page.wait_for_selector("h1:has-text('STRATEGIC WAR ROOM')")

        # Verify New Scenario Card
        print("Verifying Cyber Infrastructure Card...")
        page.wait_for_selector("#card_sim_cyber_attack")
        page.click("#card_sim_cyber_attack")

        # Verify Impact Monitor Container
        print("Verifying Impact Monitor...")
        page.wait_for_selector("#impact-monitor")
        # We check if the bar element is attached to DOM, even if width is 0
        page.wait_for_selector("#bar-tech", state="attached")

        # Initiate Simulation
        print("Initiating Cyber Simulation...")
        page.click("button:has-text('INITIATE SIMULATION')")

        # Wait for events and impact bar movement
        print("Waiting for simulation progress...")
        time.sleep(4)

        # Take Screenshot
        print(f"Taking screenshot at {SCREENSHOT_PATH}...")
        page.screenshot(path=SCREENSHOT_PATH)

        browser.close()
        print("Verification Complete.")

if __name__ == "__main__":
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    verify_enhanced_war_room()
