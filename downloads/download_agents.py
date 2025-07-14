import os
import json
import requests

def download_agents():
    """
    Downloads pre-configured agents from a remote repository.
    """
    agents_url = "https://raw.githubusercontent.com/adam-agi/adam/main/config/agents.yaml"
    response = requests.get(agents_url)
    agents_config = response.text

    with open("config/downloaded_agents.yaml", "w") as f:
        f.write(agents_config)

    print("Pre-configured agents downloaded successfully to config/downloaded_agents.yaml")

if __name__ == "__main__":
    download_agents()
