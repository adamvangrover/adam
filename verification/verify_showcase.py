from playwright.sync_api import sync_playwright


def test_showcase(page):
    page.goto("http://localhost:8000/showcase/index.html")
    page.wait_for_selector("body")
    page.screenshot(path="verification/showcase.png")

if __name__ == "__main__":
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        test_showcase(page)
        browser.close()
