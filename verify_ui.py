import os
from playwright.sync_api import sync_playwright

def verify_archive(page):
    # Navigate to the archive page
    page.goto("http://localhost:8000/showcase/market_mayhem_archive.html")

    # Wait for content to load
    page.wait_for_selector(".archive-list")

    # Screenshot
    page.screenshot(path="verification_archive.png", full_page=True)
    print("Archive screenshot captured.")

def verify_reports(page):
    # Navigate to reports page
    page.goto("http://localhost:8000/showcase/reports.html")

    # Wait for mock data to load grid
    page.wait_for_selector("#report-grid .cyber-card", timeout=5000)

    # Screenshot
    page.screenshot(path="verification_reports.png", full_page=True)
    print("Reports screenshot captured.")

if __name__ == "__main__":
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        try:
            verify_archive(page)
            verify_reports(page)
        except Exception as e:
            print(f"Error: {e}")
        finally:
            browser.close()
