import tinker
import os
from dotenv import load_dotenv
load_dotenv()
def verify_access():
    api_key = os.getenv("TINKER_API_KEY")
    if not api_key:
        print("Error: TINKER_API_KEY not found in .env file.")
        return
    try:
        service_client = tinker.ServiceClient()
        print("✅ Successfully connected to Tinker API.")
    except Exception as e:
        print(f"❌ Failed to connect. Error: {e}")
        return
    print("\n--- Available Base Models ---")
    try:
        capabilities = service_client.get_server_capabilities()
        if capabilities.supported_models:
            for item in capabilities.supported_models:
                print(f"- {item.model_name}")
        else:
            print("Could not retrieve model list.")
    except Exception as e:
        print(f"Error retrieving models: {e}")
if __name__ == "__main__":
    verify_access()
