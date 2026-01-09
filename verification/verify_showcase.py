from playwright.sync_api import sync_playwright

def verify_showcase_updates():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context()
        page = context.new_page()

        # 1. Financial Twin
        print("Verifying Financial Twin...")
        page.goto("http://localhost:8000/financial_twin.html")
        page.wait_for_selector("#risk-score")
        page.screenshot(path="verification/financial_twin.png")

        # 2. Research Lab
        print("Verifying Research Lab...")
        page.goto("http://localhost:8000/research.html")
        page.wait_for_selector("#lossChart")
        page.screenshot(path="verification/research.png")

        # 3. Trading Terminal
        print("Verifying Trading Terminal...")
        page.goto("http://localhost:8000/trading.html")
        page.wait_for_selector("#order-book")
        page.screenshot(path="verification/trading.png")

        # 4. Deployment Terminal
        print("Verifying Deployment Terminal...")
        page.goto("http://localhost:8000/deployment.html")
        page.wait_for_selector("#terminal-output")
        page.fill("#cmd-input", "help")
        page.press("#cmd-input", "Enter")
        page.screenshot(path="verification/deployment.png")

        # 5. Robo Advisor
        print("Verifying Robo Advisor...")
        page.goto("http://localhost:8000/robo_advisor.html")
        page.wait_for_selector("#currentChart")
        page.screenshot(path="verification/robo_advisor.png")

        # 6. Codex
        print("Verifying Codex...")
        page.goto("http://localhost:8000/codex.html")
        page.wait_for_selector("#file-tree")
        page.click(".tree-item") # Click first item
        page.screenshot(path="verification/codex.png")

        browser.close()

if __name__ == "__main__":
    verify_showcase_updates()
