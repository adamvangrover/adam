import time
import subprocess
import os
import sys
import json
from playwright.sync_api import sync_playwright

def verify_forward_features():
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
        print("Checking artifacts for forward view data...")
        with open("showcase/data/sovereign_artifacts/AAPL_spread.json", "r") as f:
            data = json.load(f)
            val = data["valuation"]
            assert "forward_view" in val, "Forward View data missing"
            assert "projections" in val["forward_view"], "Projections missing"
            assert len(val["forward_view"]["projections"]) == 3, "Should have 3 years of projections"
            assert "conviction_score" in val["forward_view"], "Conviction Score missing"
            assert "price_targets" in val["forward_view"], "Price Targets missing"
            assert "credit_rating" in val["risk_model"], "Credit Rating missing"
        print("Artifacts validated.")

        # 2. Verify Frontend (UI Check)
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()

            # Navigate to Dashboard
            page.goto("http://localhost:8000/showcase/sovereign_dashboard.html")

            # Wait for data load (AAPL loads by default)
            page.wait_for_selector("#convictionScore", timeout=10000)
            time.sleep(1)

            # Check Forward View Elements
            assert page.is_visible("text=FORWARD SYSTEM VIEW"), "Forward System View Header missing"

            conviction = page.inner_text("#convictionScore")
            print(f"Conviction Score: {conviction}")
            assert "/100" in conviction

            rating = page.inner_text("#creditRating")
            print(f"Credit Rating: {rating}")
            assert len(rating) > 0

            bull = page.inner_text("#bullTarget")
            base = page.inner_text("#baseTarget")
            bear = page.inner_text("#bearTarget")
            print(f"Targets: Bear={bear}, Base={base}, Bull={bull}")
            assert "$" in bull and "$" in base and "$" in bear

            # Take Screenshot
            os.makedirs("verification_screenshots", exist_ok=True)
            screenshot_path = "verification_screenshots/forward_dashboard_v1.png"
            page.screenshot(path=screenshot_path)
            print(f"Screenshot saved to {screenshot_path}")

            browser.close()

    finally:
        server_process.kill()

if __name__ == "__main__":
    verify_forward_features()
