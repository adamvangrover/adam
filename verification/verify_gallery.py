from playwright.sync_api import sync_playwright, expect

def run(playwright):
    browser = playwright.chromium.launch(headless=True)
    page = browser.new_page()
    page.set_viewport_size({"width": 1280, "height": 720})

    # Go to the gallery
    page.goto("http://localhost:8000/showcase/agent_gallery.html")

    # Wait for the grid to populate
    expect(page.locator(".agent-card").first).to_be_visible()

    # Verify Cache Status Indicator exists
    expect(page.locator("#cache-status")).to_be_visible()

    # Take screenshot of gallery
    page.screenshot(path="verification/gallery_optimized.png")
    print("Gallery screenshot taken.")

    browser.close()

with sync_playwright() as p:
    run(p)
