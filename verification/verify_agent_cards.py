from playwright.sync_api import sync_playwright
import time
import sys

def verify_agent_cards():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        try:
            # Navigate to the agents page
            page.goto("http://localhost:8000/showcase/agents.html")
        except Exception as e:
            print(f"Error connecting to server: {e}")
            sys.exit(1)

        # Wait for agents to load
        page.wait_for_selector(".agent-card")

        # --- TEST 1: Risk Assessment Agent (Existing) ---
        print("\n--- Testing Risk Assessment Agent ---")
        risk_card = page.locator(".agent-card").filter(has_text="Risk Assessment Agent").first
        if risk_card.is_visible():
            risk_card.click()
            page.wait_for_selector("#details-modal")
            time.sleep(1)
            page.screenshot(path="verification/agent_modal_risk_overview.png")

            # Close modal using the specific selector
            print("Closing modal...")
            page.locator("#details-modal button:has(.fa-times)").click()
            time.sleep(0.5)
        else:
            print("Risk Assessment Agent card not found!")

        # --- TEST 2: Code Alchemist Agent (New) ---
        print("\n--- Testing Code Alchemist Agent ---")
        # Note: Name formatting might change "Code_Alchemist_Agent" to "Code Alchemist Agent"
        # Using a partial text match is safer
        ca_card = page.locator(".agent-card").filter(has_text="Code Alchemist").first

        # Scroll to it if needed (Playwright usually handles this, but good practice)
        if ca_card.is_visible():
            ca_card.click()
            page.wait_for_selector("#details-modal")
            time.sleep(1)

            # Screenshot Tabs
            print("Screenshot: Code Alchemist Overview")
            page.screenshot(path="verification/agent_modal_ca_overview.png")

            print("Clicking Persona Tab")
            page.get_by_role("button", name="PERSONA & PROMPT").click()
            time.sleep(0.5)
            page.screenshot(path="verification/agent_modal_ca_persona.png")

            print("Clicking Logic Tab")
            page.get_by_role("button", name="LOGIC & WORKFLOW").click()
            time.sleep(0.5)
            page.screenshot(path="verification/agent_modal_ca_logic.png")

        else:
            print("Code Alchemist Agent card not found!")
            # Debug: print all texts
            print("Available cards:", page.locator(".agent-card h3").all_text_contents())

        browser.close()

if __name__ == "__main__":
    verify_agent_cards()
