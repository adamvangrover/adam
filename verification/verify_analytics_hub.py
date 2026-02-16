import os
import sys
import time
import subprocess
from playwright.sync_api import sync_playwright

def verify_analytics_hub():
    print("Starting Analytics Hub verification...")

    # Start HTTP Server
    server_process = subprocess.Popen(
        [sys.executable, "-m", "http.server", "8000"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        cwd=os.getcwd()
    )
    time.sleep(2)  # Wait for server to start

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()

            url = "http://localhost:8000/showcase/analytics_hub.html"
            print(f"Navigating to {url}")
            page.goto(url)

            # 1. Verify Header
            header = page.wait_for_selector("h1.mono")
            if header and "MARKET MAYHEM" in header.text_content():
                print("✅ Header found")
            else:
                print("❌ Header not found or incorrect")

            # 2. Verify Total Reports
            page.wait_for_selector("#stat-total-reports")
            # Wait for text to be populated (not '--')
            page.wait_for_function("document.getElementById('stat-total-reports').innerText !== '--'")
            total_reports = page.text_content("#stat-total-reports")
            print(f"✅ Total Reports loaded: {total_reports}")

            # 3. Verify Timeline Chart
            if page.is_visible("#timelineChart"):
                print("✅ Timeline Chart visible")
            else:
                print("❌ Timeline Chart not visible")

            # 4. Verify Gravity Chart
            if page.is_visible("#gravityChart"):
                print("✅ Gravity Chart visible")
            else:
                print("❌ Gravity Chart not visible")

            # 5. Verify Insights
            page.wait_for_selector("#top-insights li.insight-item")

            # Let's wait a bit for JS to render
            time.sleep(1)

            insights = page.query_selector_all("#top-insights li.insight-item")
            print(f"✅ Insights found: {len(insights)}")

            if len(insights) > 0:
                first_insight = insights[0]

                type_span = first_insight.query_selector(".insight-score")
                type_text = type_span.text_content() if type_span else "UNKNOWN"
                print(f"   First insight: {type_text}")

                title_a = first_insight.query_selector("a")
                date_span = first_insight.query_selector("span:last-child")

                if title_a:
                    print(title_a.text_content())
                if date_span:
                    print(date_span.text_content())

            # Take Screenshot
            os.makedirs("verification", exist_ok=True)
            screenshot_path = "verification/analytics_hub.png"
            page.screenshot(path=screenshot_path)
            print(f"📸 Screenshot saved to {screenshot_path}")

            browser.close()

    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        server_process.kill()

if __name__ == "__main__":
    verify_analytics_hub()
