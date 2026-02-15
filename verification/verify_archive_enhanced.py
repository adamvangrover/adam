from playwright.sync_api import sync_playwright, expect
import time

def verify_archive():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        # 1. Archive Page
        print("Navigating to Archive...")
        page.goto("http://localhost:8001/showcase/market_mayhem_archive.html")

        # Verify title
        expect(page).to_have_title("ADAM v23.5 :: MARKET MAYHEM ARCHIVE")

        # Verify presence of NEW items (Agent Note & Performance Review)
        print("Checking for Agent Note...")
        agent_note = page.locator(".archive-item").filter(has_text="Agent Alignment Log").first
        expect(agent_note).to_be_visible()
        # Verify Color (Green for Agent Note)
        expect(agent_note.locator(".type-badge")).to_have_css("background-color", "rgb(0, 255, 0)")

        print("Checking for Performance Review...")
        perf_review = page.locator(".archive-item").filter(has_text="Model Performance Review").first
        expect(perf_review).to_be_visible()
        # Verify Color (Blue for Performance Review)
        expect(perf_review.locator(".type-badge")).to_have_css("background-color", "rgb(96, 165, 250)") # #60a5fa

        page.screenshot(path="verification/archive_enhanced.png")
        print("Enhanced Archive screenshot saved.")

        # 2. Click Agent Note
        agent_note.locator("a.cyber-btn").click()

        # 3. Verify Agent Note Page
        print("Verifying Agent Note Page...")
        expect(page).to_have_title("ADAM v23.5 :: Agent Alignment Log: Protocol v23.5 (December 12, 2025)")
        expect(page.locator("h1.title")).to_contain_text("Agent Alignment Log")
        # Be specific to avoid matching hidden raw data
        expect(page.locator("h2", has_text="Neuro-Symbolic Bridge")).to_be_visible()

        page.screenshot(path="verification/agent_note.png")
        print("Agent Note screenshot saved.")

        browser.close()

if __name__ == "__main__":
    verify_archive()
