import time
import subprocess
import os
import sys
import json
from playwright.sync_api import sync_playwright

def verify_valuation_features():
    # Start HTTP Server
    server_process = subprocess.Popen(
        [sys.executable, "-m", "http.server", "8000"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        cwd=os.getcwd()
    )
    time.sleep(2) # Wait for server to start

    try:
        # 1. Verify Artifact Data Structure (Backend Check)
        print("Checking artifacts for valuation data...")
        with open("showcase/data/sovereign_artifacts/AAPL_spread.json", "r") as f:
            data = json.load(f)
            assert "valuation" in data, "Valuation data missing from artifact"
            assert "dcf" in data["valuation"], "DCF data missing"
            assert "risk_model" in data["valuation"], "Risk Model data missing"
            assert data["valuation"]["risk_model"]["z_score"] > 0, "Z-Score invalid"
        print("Artifacts validated.")

        # 2. Verify Frontend (UI Check)
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()

            # Listen for console logs
            page.on("console", lambda msg: print(f"PAGE LOG: {msg.text}"))
            page.on("pageerror", lambda err: print(f"PAGE ERROR: {err}"))

            # Navigate to Dashboard
            page.goto("http://localhost:8000/showcase/sovereign_dashboard.html")

            # Wait for data load (AAPL loads by default)
            # Increased timeout
            page.wait_for_selector("#dcfSharePrice", timeout=10000)
            time.sleep(2)

            # Check Panel Title
            assert page.is_visible("text=VALUATION & RISK"), "Valuation Panel not found"

            # Check DCF Values
            share_price = page.inner_text("#dcfSharePrice")
            print(f"Initial Share Price: {share_price}")
            assert "$" in share_price

            # Check Sliders
            assert page.is_visible("#waccSlider"), "WACC Slider missing"
            assert page.is_visible("#growthSlider"), "Growth Slider missing"

            # Check Risk Gauges
            assert page.is_visible("text=PROBABILITY OF DEFAULT"), "PD Label missing"
            assert page.is_visible("text=LOSS GIVEN DEFAULT"), "LGD Label missing"

            # Test Interactivity: Move Slider
            # Wait for initial calculation
            page.wait_for_function("document.getElementById('dcfSharePrice').innerText !== '$--.--'")
            initial_val = page.inner_text("#dcfEnterpriseValue")
            print(f"Initial EV: {initial_val}")

            # Change WACC (Increase WACC -> Lower Valuation)
            # Use evaluate for range input
            page.evaluate("document.getElementById('waccSlider').value = 12.0")
            page.dispatch_event("#waccSlider", "input") # Trigger event

            time.sleep(1)
            new_val = page.inner_text("#dcfEnterpriseValue")
            print(f"Value after WACC increase: {new_val}")

            assert initial_val != new_val, "Slider did not update valuation"

            # Check AI Outlook
            assert page.is_visible("#aiOutlook"), "AI Outlook missing"
            outlook_text = page.inner_text("#aiOutlook")
            assert len(outlook_text) > 20, "Outlook text too short"
            print("AI Outlook verified.")

            # Take Screenshot
            os.makedirs("verification_screenshots", exist_ok=True)
            screenshot_path = "verification_screenshots/valuation_dashboard_v1.png"
            page.screenshot(path=screenshot_path)
            print(f"Screenshot saved to {screenshot_path}")

            browser.close()

    finally:
        server_process.kill()

if __name__ == "__main__":
    verify_valuation_features()
