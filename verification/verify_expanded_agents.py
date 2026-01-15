from playwright.sync_api import sync_playwright
import time

def verify_expanded_agents():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        # Navigate to the agents page
        page.goto("http://localhost:8000/agents.html")

        # Wait for agents to load
        page.wait_for_selector(".agent-card")

        # Count agents
        count = page.locator(".agent-card").count()
        print(f"Found {count} agents in the grid.")

        if count < 10:
            print("FAILED: Expected at least 10 agents.")
            browser.close()
            return

        # Check for specific new agents
        new_agents = ["Technical Analyst Agent", "Red Team Agent", "Macroeconomic Analysis Agent"]
        for agent_name in new_agents:
            if page.locator(".agent-card").filter(has_text=agent_name).count() > 0:
                print(f"Verified presence of: {agent_name}")
            else:
                print(f"FAILED: Could not find {agent_name}")

        # Click on Red Team Agent to check details
        red_team_card = page.locator(".agent-card").filter(has_text="Red Team Agent").first
        if red_team_card.is_visible():
            print("Clicking Red Team Agent...")
            red_team_card.click()
            time.sleep(1)

            # Verify specific detail content
            modal_text = page.locator("#details-modal").inner_text()
            if "Adversarial Self-Correction Loop" in modal_text or "RedTeamAgent" in modal_text:
                print("Verified Red Team Agent details content.")
            else:
                print("FAILED: Red Team Agent details content mismatch.")

            page.screenshot(path="verification/red_team_modal.png")

        browser.close()

if __name__ == "__main__":
    verify_expanded_agents()
