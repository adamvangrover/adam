from playwright.sync_api import sync_playwright
import time
import sys

def verify_expanded_roster():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        # Navigate to the agents page
        try:
            page.goto("http://localhost:8000/showcase/agents.html")
        except Exception as e:
            print(f"Error connecting to server: {e}")
            sys.exit(1)

        # Wait for agents to load
        try:
            page.wait_for_selector(".agent-card", timeout=5000)
        except:
            print("Timeout waiting for agent cards. Check if JS loaded correctly.")
            browser.close()
            sys.exit(1)

        # Count total agents
        cards = page.locator(".agent-card")
        count = cards.count()
        print(f"Total Agent Cards Found: {count}")

        # Expected new agents to verify
        expected_agents = [
            "Code Alchemist Agent",
            "News Bot Agent",
            "Supply Chain Risk Agent",
            "Regulatory Compliance Agent",
            "AI Portfolio Optimizer"
        ]

        missing_agents = []
        found_agents = []

        all_card_texts = cards.all_text_contents()
        # Normalize text for checking
        all_card_texts = [t.lower() for t in all_card_texts]

        for agent in expected_agents:
            # Simple check if any card text contains the agent name
            # Note: The UI might format names differently (e.g. underscores to spaces)
            # My JS formatter does that. "Code_Alchemist_Agent" -> "Code Alchemist Agent"

            # Helper to match roughly
            match = False
            for text in all_card_texts:
                if agent.lower() in text:
                    match = True
                    break

            if match:
                found_agents.append(agent)
                print(f"✅ Found: {agent}")
            else:
                missing_agents.append(agent)
                print(f"❌ Missing: {agent}")

        browser.close()

        if missing_agents:
            print(f"\nVerification FAILED. Missing agents: {missing_agents}")
            sys.exit(1)
        else:
            print("\nVerification SUCCESS. All expected agents found.")

if __name__ == "__main__":
    verify_expanded_roster()
