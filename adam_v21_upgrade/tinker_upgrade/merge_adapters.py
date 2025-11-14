import tinker
import os
import asyncio
from dotenv import load_dotenv

load_dotenv()

# --- Configuration ---
SOUL_ADAPTER_PATH = "adam_aligned_soul_v1"
FINAL_ADAPTER_PATH = "adam_final_agent_lora"

async def main():
    """
    Asynchronously renames the final 'Soul' adapter to its production-ready
    designation, 'adam_final_agent_lora'.
    """
    print("--- [Final Step] Starting Adapter Merge Process ---")
    print(f"Designating '{SOUL_ADAPTER_PATH}' as the final production adapter...")

    try:
        service_client = tinker.ServiceClient()
        rest_client = await service_client.create_rest_client_async()

        print(f"Renaming '{SOUL_ADAPTER_PATH}' to '{FINAL_ADAPTER_PATH}' for production deployment...")

        # Asynchronously rename the checkpoint in cloud storage
        future = await rest_client.rename_checkpoint_async(
            path_from=SOUL_ADAPTER_PATH,
            path_to=FINAL_ADAPTER_PATH
        )
        await future

        print(f"✅ Successfully created final merged adapter: '{FINAL_ADAPTER_PATH}'")
        print("This final, composed adapter is now saved in Tinker cloud storage.")

    except Exception as e:
        print(f"❌ An error occurred during the final adapter creation process: {e}")
        print("Please ensure that the 'Soul' adapter was trained successfully.")
        exit(1)

if __name__ == "__main__":
    asyncio.run(main())
