import os
import time
import subprocess
from playwright.sync_api import sync_playwright

# Configuration
PORT = 8000
DIRECTORY = "showcase"  # The folder where your HTML files are located
SCREENSHOT_DIR = "verification"

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def verify_ui():
    ensure_dir(SCREENSHOT_DIR)
    
    print(f"Starting local server on port {PORT} serving '{DIRECTORY}'...")
    # Start a static file server in the background
    # We use subprocess to keep it independent of the test thread
    server_process = subprocess.Popen(
        ["python3", "-m", "http.server", str(PORT)], 
        cwd=DIRECTORY
    )
    
    time.sleep(2)  # Give server time to spin up

    try:
        with sync_playwright() as p:
            print("Launching Browser...")
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()

            # 1. Verify Neural Dashboard
            # --------------------------
            print("1. Verifying Neural Dashboard...")
            try:
                page.goto(f"http://localhost:{PORT}/neural_dashboard.html")
                # Wait for the specific graph canvas ID
                page.wait_for_selector("#viz-canvas", timeout=5000)
                # Allow animations (pulse/logs) to run briefly
                time.sleep(2) 
                page.screenshot(path=f"{SCREENSHOT_DIR}/1_neural_dashboard.png")
                print("   -> Success")
            except Exception as e:
                print(f"   -> Failed: {e}")

            # 2. Verify Deep Dive Explorer
            # --------------------------
            print("2. Verifying Deep Dive Explorer...")
            try:
                page.goto(f"http://localhost:{PORT}/deep_dive.html")
                # Wait for sidebar items to populate
                page.wait_for_selector("#report-list div", timeout=5000)
                
                # Simulate user interaction: Click the first report
                page.click("#report-list > div:first-child")
                
                # Wait for the charts (phase-widgets) to render
                page.wait_for_selector(".phase-widget", timeout=5000)
                time.sleep(1)
                page.screenshot(path=f"{SCREENSHOT_DIR}/2_deep_dive.png")
                print("   -> Success")
            except Exception as e:
                print(f"   -> Failed: {e}")

            # 3. Verify Financial Digital Twin
            # --------------------------
            print("3. Verifying Financial Twin...")
            try:
                page.goto(f"http://localhost:{PORT}/financial_twin.html")
                # Wait for Vis.js network container
                page.wait_for_selector("#network-container", timeout=5000)
                time.sleep(2) # Wait for physics stabilization
                page.screenshot(path=f"{SCREENSHOT_DIR}/3_financial_twin.png")
                print("   -> Success")
            except Exception as e:
                print(f"   -> Failed: {e}")

            # 4. Verify Agent Registry
            # --------------------------
            print("4. Verifying Agent Registry...")
            try:
                page.goto(f"http://localhost:{PORT}/agents.html")
                # Wait for the glass cards
                page.wait_for_selector(".glass-panel", timeout=5000)
                
                # Simulate opening the terminal
                page.click("#terminal") 
                time.sleep(1)
                
                page.screenshot(path=f"{SCREENSHOT_DIR}/4_agent_registry.png")
                print("   -> Success")
            except Exception as e:
                print(f"   -> Failed: {e}")

            browser.close()
            print(f"\nVerification complete. Screenshots saved to '{SCREENSHOT_DIR}/'")

    except Exception as e:
        print(f"CRITICAL ERROR: {e}")

    finally:
        # Always kill the server, even if tests fail
        print("Shutting down server...")
        server_process.terminate()

if __name__ == "__main__":
    verify_ui()