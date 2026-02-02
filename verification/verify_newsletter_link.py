from playwright.sync_api import sync_playwright
import os

def test_newsletter_archive_link():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        # Load the archive page
        archive_path = os.path.abspath("showcase/market_mayhem_archive.html")
        page.goto(f"file://{archive_path}")

        # Look for the new entry
        new_entry_link = page.locator("a[href='newsletter_market_mayhem_jan_31_2026.html']")

        # Screenshot the archive focusing on the top of the grid
        page.locator("#archiveGrid").screenshot(path="verification/archive_with_new_entry.png")
        print("Archive screenshot captured.")

        if new_entry_link.count() > 0:
             print("PASS: New newsletter link found in archive.")
             new_entry_link.click()
             page.wait_for_load_state('networkidle')

             # Verify we are on the newsletter page
             page.screenshot(path="verification/newsletter_page.png")
             print("Newsletter page screenshot captured.")

             if "Jan 31, 2026" in page.title():
                 print("PASS: Newsletter page loaded correctly.")
             else:
                 print("FAIL: Newsletter title mismatch.")
        else:
             print("FAIL: New newsletter link NOT found in archive.")

        browser.close()

if __name__ == "__main__":
    test_newsletter_archive_link()
