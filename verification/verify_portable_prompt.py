from playwright.sync_api import sync_playwright
import time
import sys

def verify_portable_prompt():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        try:
            page.goto("http://localhost:8000/showcase/agents.html")
        except Exception as e:
            print(f"Error connecting to server: {e}")
            sys.exit(1)

        page.wait_for_selector(".agent-card")

        # Test Code Alchemist Agent
        print("\n--- Testing Code Alchemist Agent ---")
        ca_card = page.locator(".agent-card").filter(has_text="Code Alchemist").first

        if ca_card.is_visible():
            ca_card.click()
            page.wait_for_selector("#details-modal")
            time.sleep(1)

            # Go to Persona Tab
            print("Clicking Persona Tab")
            page.get_by_role("button", name="PERSONA & PROMPT").click()
            time.sleep(0.5)

            # Take screenshot of the specific section
            print("Screenshot: Portable Prompt Section")
            page.screenshot(path="verification/agent_modal_portable_prompt.png")

            # Verify the text content starts with ### SYSTEM PROMPT
            prompt_content = page.locator("#modal-md-copy").inner_text()
            if "### SYSTEM PROMPT: Code_Alchemist_Agent ###" in prompt_content:
                print("✅ Portable prompt loaded correctly.")
            else:
                print("❌ Portable prompt mismatch or missing!")
                print(f"Content start: {prompt_content[:50]}...")

        else:
            print("Code Alchemist Agent card not found!")

        browser.close()

if __name__ == "__main__":
    verify_portable_prompt()
