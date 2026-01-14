from playwright.sync_api import sync_playwright, expect
import time

def verify_archive():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        # Verify Archive Page
        print("Navigating to Archive Page...")
        page.goto("http://localhost:8000/showcase/market_mayhem_archive.html")
        expect(page).to_have_title("ADAM v23.5 :: MARKET MAYHEM ARCHIVE")

        # Verify specific elements from Origin (Search bar)
        expect(page.locator("#searchInput")).to_be_visible()
        expect(page.locator("#yearFilter")).to_be_visible()

        # Verify elements from HEAD (Dec 02 2025)
        # Note: HEAD items were inserted.
        # "THE GREAT CALIBRATION" is Dec 02
        expect(page.get_by_text("THE GREAT CALIBRATION")).to_be_visible()

        # Verify elements from Origin (Liquidity Trap - Sep 2025)
        expect(page.get_by_text("THE LIQUIDITY TRAP")).to_be_visible()

        page.screenshot(path="verification/archive_page.png", full_page=True)
        print("Archive page verified and screenshot saved.")

        # Verify a Monthly Newsletter (Origin Style - Paper Sheet)
        print("Navigating to July 2025 (Monthly)...")
        page.goto("http://localhost:8000/showcase/newsletter_market_mayhem_jul_2025.html")
        # Check for Paper Sheet class
        expect(page.locator(".paper-sheet")).to_be_visible()
        page.screenshot(path="verification/jul_2025_monthly.png", full_page=True)
        print("July 2025 verified.")

        # Verify a Weekly Newsletter (HEAD Style - Cyber Badge)
        print("Navigating to Dec 02 2025 (Weekly)...")
        page.goto("http://localhost:8000/showcase/newsletter_market_mayhem_dec_02_2025.html")
        # Check for Cyber Badge
        expect(page.locator(".cyber-badge-overlay")).to_be_visible()
        page.screenshot(path="verification/dec_02_2025_weekly.png", full_page=True)
        print("Dec 02 2025 verified.")

        browser.close()

if __name__ == "__main__":
    verify_archive()
