import os
from playwright.sync_api import sync_playwright

def verify_archive():
    print("Verifying Market Mayhem Deep Archive...")

    archive_path = os.path.abspath("showcase/market_mayhem_archive.html")
    if not os.path.exists(archive_path):
        print(f"FAILED: {archive_path} does not exist.")
        return

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(f"file://{archive_path}")

        # 1. Check Title
        title = page.title()
        if "MARKET MAYHEM ARCHIVE" in title:
            print(f"PASSED: Title is '{title}'")
        else:
            print(f"FAILED: Title mismatch: '{title}'")

        # 2. Check Chart Canvas
        if page.is_visible("#sentimentChart"):
            print("PASSED: Sentiment Chart canvas is visible.")
        else:
            print("FAILED: Sentiment Chart canvas not found.")

        # 3. Check Filters
        if page.is_visible("#yearFilter") and page.is_visible("#typeFilter"):
            print("PASSED: Facet filters are visible.")
        else:
            print("FAILED: Facet filters missing.")

        # 4. Check Archive Grid
        items = page.query_selector_all(".archive-item")
        if len(items) > 0:
            print(f"PASSED: Found {len(items)} archive items.")
        else:
            print("FAILED: No archive items found.")

        # 5. Check Deep Link to Report and Cyber-HUD
        # We'll just load a known report file if it exists
        report_link = items[0].get_attribute("href")
        if report_link:
            print(f"Navigate to report: {report_link}")
            page.goto(f"file://{os.path.abspath(os.path.join('showcase', report_link))}")

            if page.is_visible(".cyber-sidebar"):
                print("PASSED: Cyber-HUD Sidebar found in report.")
            else:
                print("FAILED: Cyber-HUD Sidebar not found.")

            if page.is_visible(".sidebar-widget"):
                print("PASSED: Sidebar widgets found.")

        browser.close()

if __name__ == "__main__":
    verify_archive()
