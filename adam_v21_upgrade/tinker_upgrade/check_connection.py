import tinker
import os
import asyncio
from dotenv import load_dotenv

load_dotenv()

async def main():
    """
    Asynchronously verifies the connection to the Tinker API and lists
    available base models.
    """
    api_key = os.getenv("TINKER_API_KEY")
    if not api_key:
        print("Error: TINKER_API_KEY not found in .env file.")
        return

    try:
        service_client = tinker.ServiceClient()
        # A simple async call to verify the connection
        await service_client.get_server_capabilities_async()
        print("✅ Successfully connected to Tinker API.")
    except Exception as e:
        print(f"❌ Failed to connect. Error: {e}")
        return

    print("\\n--- Available Base Models ---")
    try:
        capabilities = await service_client.get_server_capabilities_async()
        if capabilities.supported_models:
            for item in capabilities.supported_models:
                print(f"- {item.model_name}")
        else:
            print("Could not retrieve model list.")
    except Exception as e:
        print(f"Error retrieving models: {e}")

if __name__ == "__main__":
    asyncio.run(main())
