import time
import subprocess
import os
import sys
import json
from playwright.sync_api import sync_playwright

def verify_reports():
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
        print("Checking artifacts for report rationale...")
        with open("showcase/data/sovereign_artifacts/AAPL_spread.json", "r") as f:
            data = json.load(f)
            assert "rationale" in data["valuation"]["risk_model"], "Risk Rationale missing"
            assert "rationale" in data["valuation"]["forward_view"], "Valuation Rationale missing"
            print(f"Risk Rationale: {data['valuation']['risk_model']['rationale'][:50]}...")
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
            page.wait_for_selector("#convictionScore", timeout=10000)
            time.sleep(1)

            # Click "VIEW REPORTS"
            page.click("text=VIEW REPORTS")

            # Check Overlay Visibility
            page.wait_for_selector("#reportOverlay.active")
            print("Report Overlay is active.")

            # Check Report Content
            risk_text = page.inner_text("#riskReportText")
            valuation_text = page.inner_text("#valuationReportText")

            print(f"Risk Report Content: {risk_text[:50]}...")
            assert "Z-Score" in risk_text
            assert "Equity Price Target" in valuation_text

            # Test Edit
            page.fill("#riskReportText", "Edited Risk Report Content.")

            # Check Download Button exists (can't easily verify download in headless without complex setup)
            assert page.is_visible("text=DOWNLOAD JSON")
            assert page.is_visible("text=UPLOAD JSON")

            # Take Screenshot of Overlay
            os.makedirs("verification_screenshots", exist_ok=True)
            screenshot_path = "verification_screenshots/reports_overlay_v1.png"
            page.screenshot(path=screenshot_path)
            print(f"Screenshot saved to {screenshot_path}")

            browser.close()

    finally:
        server_process.kill()

if __name__ == "__main__":
    verify_reports()
