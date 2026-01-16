from playwright.sync_api import sync_playwright
import os

def verify_archive():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        # Verify Archive Page
        archive_path = os.path.abspath("showcase/market_mayhem_archive.html")
        page.goto(f"file://{archive_path}")
        print(f"Loaded {archive_path}")

        # Check title
        assert "MARKET MAYHEM ARCHIVE" in page.title()

        # Check for 2025 header
        page.locator(".year-header", has_text="2025 ARCHIVE").wait_for()

        # Check for specific report
        page.locator(".archive-item", has_text="Nvidia Corporation (NVDA) Report").first.wait_for()

        page.screenshot(path="verification_images/archive_page.png")
        print("Screenshot saved to verification_images/archive_page.png")

        # Verify Report Page
        report_path = os.path.abspath("showcase/nvda_company_report_20250226.html")
        page.goto(f"file://{report_path}")
        print(f"Loaded {report_path}")

        # Check title
        assert "Nvidia Corporation (NVDA) Report" in page.locator("h1.title").text_content()

        # Check content
        assert "Executive Summary" in page.content()
        assert "315" in page.content() # Price target

        page.screenshot(path="verification_images/report_page.png")
        print("Screenshot saved to verification_images/report_page.png")

        browser.close()

if __name__ == "__main__":
    verify_archive()
