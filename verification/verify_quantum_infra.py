import os
from playwright.sync_api import sync_playwright

def verify_quantum_infra():
    # Get absolute path to the file
    cwd = os.getcwd()
    file_path = f"file://{cwd}/showcase/quantum_infrastructure.html"

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        print(f"Navigating to {file_path}")
        page.goto(file_path)

        # Wait for logs to be populated (indicates JS ran and data loaded)
        try:
            page.wait_for_selector(".log-entry", timeout=5000)
            print("Log entries found.")
        except:
            print("Timeout waiting for log entries. Checking console logs...")

        # Take screenshot
        output_path = "verification/quantum_infra.png"
        page.screenshot(path=output_path, full_page=True)
        print(f"Screenshot saved to {output_path}")

        browser.close()

if __name__ == "__main__":
    verify_quantum_infra()
