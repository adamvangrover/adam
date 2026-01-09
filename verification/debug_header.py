from playwright.sync_api import sync_playwright

def debug_header():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        try:
            page.goto("http://localhost:3000", timeout=30000)
            page.wait_for_selector("header")
            page.screenshot(path="verification/debug_screenshot.png")
            print("Debug screenshot taken.")

            # Print the HTML of the header
            header_html = page.eval_on_selector("header", "el => el.outerHTML")
            print("Header HTML:", header_html)

        except Exception as e:
            print(f"Debug failed: {e}")

        finally:
            browser.close()

if __name__ == "__main__":
    debug_header()
