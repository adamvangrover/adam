
import os
import subprocess
import time
from playwright.sync_api import sync_playwright

def verify_ui():
    # Start a static file server in the background
    server_process = subprocess.Popen(["python3", "-m", "http.server", "8000"], cwd="showcase")
    time.sleep(2)  # Wait for server to start

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()

            # Capture console logs
            page.on("console", lambda msg: print(f"PAGE LOG: {msg.text}"))
            page.on("pageerror", lambda err: print(f"PAGE ERROR: {err}"))

            # Verify Neural Dashboard
            print("Verifying Neural Dashboard...")
            page.goto("http://localhost:8000/neural_dashboard.html")
            try:
                page.wait_for_selector("#viz-canvas", timeout=5000)
                time.sleep(2)
                page.screenshot(path="verification/neural_dashboard.png")
            except Exception as e:
                print(f"Error verifying Neural Dashboard: {e}")
                page.screenshot(path="verification/error_neural.png")

            # Verify Deep Dive
            print("Verifying Deep Dive Explorer...")
            page.goto("http://localhost:8000/deep_dive.html")
            try:
                # Wait for sidebar items
                page.wait_for_selector("#report-list div", timeout=5000)
                page.screenshot(path="verification/deep_dive_sidebar.png")

                # Click first report in sidebar
                page.click("#report-list > div:first-child")

                # Now wait for report view widgets
                page.wait_for_selector(".phase-widget", timeout=5000)
                time.sleep(1)
                page.screenshot(path="verification/deep_dive_report.png")
            except Exception as e:
                print(f"Error verifying Deep Dive: {e}")
                page.screenshot(path="verification/error_deep_dive.png")

            # Verify Financial Twin
            print("Verifying Financial Twin...")
            page.goto("http://localhost:8000/financial_twin.html")
            try:
                page.wait_for_selector("#network-container", timeout=5000)
                time.sleep(1)
                page.screenshot(path="verification/financial_twin.png")
            except Exception as e:
                print(f"Error verifying Financial Twin: {e}")
                page.screenshot(path="verification/error_twin.png")

            # Verify Agents
            print("Verifying Agent Registry...")
            page.goto("http://localhost:8000/agents.html")
            try:
                page.wait_for_selector(".glass-panel", timeout=5000)
                page.screenshot(path="verification/agents.png")
            except Exception as e:
                print(f"Error verifying Agents: {e}")
                page.screenshot(path="verification/error_agents.png")

            print("Screenshots saved to verification/")
            browser.close()

    finally:
        server_process.terminate()

if __name__ == "__main__":
    verify_ui()
