from playwright.sync_api import sync_playwright
import time
import os

OUTPUT_DIR = "verification"
SCREENSHOT_PATH = os.path.join(OUTPUT_DIR, "war_room_comprehensive_verification.png")

def verify_comprehensive_war_room():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={'width': 1280, 'height': 800})

        # Navigate to War Room
        print("Navigating to War Room...")
        page.goto("http://localhost:8080/showcase/war_room.html")

        # Wait for page load
        page.wait_for_selector("h1:has-text('STRATEGIC WAR ROOM')")

        # Verify Quantum Scenario Card
        print("Verifying Quantum Decryption Card...")
        page.wait_for_selector("#card_sim_quantum_event")
        page.click("#card_sim_quantum_event")

        # Verify Impact Monitor (Full Grid)
        print("Verifying Impact Monitor Grid...")
        page.wait_for_selector("#impact-monitor")
        # Check for new sectors - ensure elements exist, using state="attached" to avoid visibility timeouts if partially hidden or animating
        page.wait_for_selector("text=POLITICAL", state="attached")
        page.wait_for_selector("text=SHADOW BANKING", state="attached")

        # Initiate Simulation
        print("Initiating Quantum Simulation...")
        page.click("button:has-text('INITIATE SIMULATION')")

        # Wait for simulation to run partially
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
    verify_comprehensive_war_room()
