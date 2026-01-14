from playwright.sync_api import sync_playwright
import os

def run():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        # 1. Load Agents Page
        agents_url = f"file://{os.getcwd()}/showcase/agents.html"
        print(f"Navigating to {agents_url}")
        page.goto(agents_url)

        # 2. Click WORKFLOWS filter
        print("Clicking WORKFLOWS filter...")
        workflow_btn = page.get_by_role("button", name="WORKFLOWS")
        if workflow_btn.is_visible():
            workflow_btn.click()
            page.wait_for_timeout(500)

        # Debug: Print all h3 texts
        print("Visible Agent Names:")
        for h3 in page.locator("h3").all():
            print(f"- '{h3.text_content().strip()}'")

        # 3. Find SNC Workflow Card
        # Try matching partial text "SNC"
        snc_card = page.locator("h3", has_text="SNC")

        if snc_card.count() > 0:
            print("SNC Card found. Clicking...")
            snc_card.first.click()

            # 4. Verify Navigation
            page.wait_for_timeout(1000)
            print(f"Current URL: {page.url}")

            if "snc_cover.html" in page.url:
                print("Navigation successful!")
                page.wait_for_load_state("domcontentloaded")

                header = page.locator("h2", has_text="SNC Regulatory Cover Sheet")
                if header.is_visible():
                    print("SNC Cover Page loaded correctly.")
                    page.screenshot(path="verification/snc_cover_page.png")
                else:
                    print("SNC Cover Page header not found.")
            else:
                print("Navigation failed.")
        else:
            print("SNC Analyst Workflow card not found.")

        browser.close()

if __name__ == "__main__":
    run()
