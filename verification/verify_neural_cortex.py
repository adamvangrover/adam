from playwright.sync_api import sync_playwright
import os

def run():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        # Capture console logs
        page.on("console", lambda msg: print(f"Console: {msg.text}"))
        page.on("pageerror", lambda exc: print(f"Page Error: {exc}"))

        # Get absolute path to the file
        cwd = os.getcwd()
        file_path = os.path.join(cwd, "core/agents/architect_agent/index.html")
        url = f"file://{file_path}"

        print(f"Navigating to: {url}")
        try:
            page.goto(url)
            # Wait for the title to be correct
            page.wait_for_selector("text=NEURAL_CORTEX_VIEW", timeout=5000)
        except Exception as e:
            print(f"Navigation/Wait failed: {e}")

        # Take screenshot regardless
        output_path = "verification/neural_cortex.png"
        page.screenshot(path=output_path)
        print(f"Screenshot saved to {output_path}")

        browser.close()

if __name__ == "__main__":
    if not os.path.exists("verification"):
        os.makedirs("verification")
    run()
