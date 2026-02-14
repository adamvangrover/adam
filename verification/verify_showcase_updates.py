from playwright.sync_api import sync_playwright
import time

def verify(page):
    page.on("console", lambda msg: print(f"CONSOLE: {msg.text}"))
    page.on("pageerror", lambda err: print(f"PAGE ERROR: {err}"))

    # 2. Sovereign Dashboard
    print("Verifying Sovereign Dashboard...")
    page.goto("http://localhost:8080/showcase/sovereign_dashboard.html")
    page.wait_for_load_state("networkidle")
    time.sleep(2)

    # Check for News Feed
    news_feed = page.locator("#newsFeed")
    if news_feed.is_visible():
        print("PASS: News Feed visible")
        # Check content
        try:
            content = news_feed.inner_text().lower()
            if "sector rotation" in content:
                 print("PASS: News Feed populated")
            else:
                 print(f"FAIL: News Feed empty or static. Content: '{content}'")
        except Exception as e:
            print(f"FAIL: Could not read news feed: {e}")
    else:
        print("FAIL: News Feed missing")

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page()
    verify(page)
    browser.close()
