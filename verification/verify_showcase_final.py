from playwright.sync_api import sync_playwright
import os

def run():
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()

        cwd = os.getcwd()
        url = f"file://{cwd}/showcase/reports.html"
        print(f"Navigating to {url}")
        page.goto(url)
        page.wait_for_timeout(2000)

        output_path = f"{cwd}/verification/showcase_reports_final.png"
        page.screenshot(path=output_path)
        print(f"Screenshot saved to {output_path}")

        browser.close()

if __name__ == "__main__":
    run()
