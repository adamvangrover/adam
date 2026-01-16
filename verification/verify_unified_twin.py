import os
import sys
import asyncio
from playwright.async_api import async_playwright

async def run():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        # Determine the file path
        file_path = os.path.abspath("showcase/financial_twin.html")
        url = f"file://{file_path}"

        print(f"Navigating to {url}...")
        await page.goto(url)

        # 1. Verify Title
        title = await page.title()
        print(f"Page Title: {title}")
        if "Unified Banking World Model" not in title:
            print("❌ Title mismatch.")
            sys.exit(1)

        # 2. Verify System Brain Link
        brain_link = page.locator("a[href='system_brain.html']")
        if await brain_link.count() > 0:
            print("✅ System Brain link found.")
        else:
            print("❌ System Brain link missing.")
            sys.exit(1)

        # 3. Verify AI Grid Filter
        ai_filter = page.locator("button[data-group='ai_infra']")
        if await ai_filter.count() > 0:
            print("✅ AI Grid filter button found.")
        else:
            print("❌ AI Grid filter button missing.")
            sys.exit(1)

        # 4. Verify Canvas (Vis.js Network)
        canvas = page.locator("#twin-network canvas")
        await canvas.wait_for(state="visible", timeout=10000)
        print("✅ Network canvas visible.")

        # Take a screenshot
        os.makedirs("verification_artifacts", exist_ok=True)
        await page.screenshot(path="verification_artifacts/unified_twin_verified.png")
        print("📸 Screenshot saved to verification_artifacts/unified_twin_verified.png")

        await browser.close()

if __name__ == "__main__":
    asyncio.run(run())
