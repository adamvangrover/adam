import time
import subprocess
import os
import sys
import json
from playwright.sync_api import sync_playwright

def verify_full_report_suite():
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
        print("Checking artifacts for full report data...")
        with open("showcase/data/sovereign_artifacts/AAPL_report.json", "r") as f:
            data = json.load(f)
            assert "scenarios" in data, "Scenarios missing"
            assert len(data["scenarios"]) == 3, "Should have 3 scenarios"
            assert "swot" in data, "SWOT missing"
            assert "cap_structure" in data, "Cap Structure missing"
            assert "citations" in data, "Citations missing"
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

            # Open Reports
            page.click("text=VIEW REPORTS")
            page.wait_for_selector("#reportOverlay.active")
            print("Report Overlay Opened.")

            # Check Executive Tab
            assert page.is_visible("#swotStrengths"), "SWOT Strengths missing"
            assert page.is_visible("#citationsList"), "Citations List missing"
            print("Executive Tab Verified.")

            # Switch to Credit Tab
            page.click("text=CREDIT")
            time.sleep(0.5)
            assert page.is_visible("text=INTERACTIVE Z-SCORE CALCULATOR"), "Z-Score Calculator missing"
            assert page.is_visible("#capStructureTable"), "Cap Structure Table missing"
            print("Credit Tab Verified.")

            # Test Z-Score Interaction
            page.fill("#inputZ_A", "0.5") # Working Capital / Assets
            page.dispatch_event("#inputZ_A", "input")
            z_score = page.inner_text("#calcZScore")
            print(f"Calculated Z-Score: {z_score}")
            assert z_score != "--", "Z-Score calculation failed"

            # Switch to Equity Tab
            page.click("text=EQUITY")
            time.sleep(0.5)
            assert page.is_visible("#scenarioChart"), "Scenario Chart Canvas missing"
            print("Equity Tab Verified.")

            # Take Screenshot
            os.makedirs("verification_screenshots", exist_ok=True)
            screenshot_path = "verification_screenshots/full_report_suite_v1.png"
            page.screenshot(path=screenshot_path)
            print(f"Screenshot saved to {screenshot_path}")

            browser.close()

    finally:
        server_process.kill()

if __name__ == "__main__":
    verify_full_report_suite()
