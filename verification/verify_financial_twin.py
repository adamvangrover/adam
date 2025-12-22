from playwright.sync_api import sync_playwright
import os

def run():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        # Get absolute path to the file
        cwd = os.getcwd()
        file_path = os.path.join(cwd, "showcase/financial_twin.html")
        url = f"file://{file_path}"

        print(f"Navigating to: {url}")
        page.goto(url)

        # Wait for title
        page.wait_for_selector("text=ADAM")
        page.wait_for_selector("text=Risk Topography")

        # Click "Inject Stress" button to test interactivity
        page.click("#btn-simulate")

        # Wait a bit for simulation
        page.wait_for_timeout(2000)

        # Take screenshot
        output_path = "verification/financial_twin.png"
        page.screenshot(path=output_path)
        print(f"Screenshot saved to {output_path}")

        browser.close()

if __name__ == "__main__":
    if not os.path.exists("verification"):
        os.makedirs("verification")
    run()
