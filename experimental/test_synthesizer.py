import subprocess
import time
from playwright.sync_api import sync_playwright

def verify_synthesizer():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        # Test War Room static page
        page.goto("file:///app/showcase/war_room_v2.html")
        page.wait_for_selector("text=The War Room // SYNTHESIZER V2")
        page.screenshot(path="verification_war_room.png", full_page=True)
        print("Successfully verified War Room static page!")

        browser.close()

if __name__ == "__main__":
    verify_synthesizer()
