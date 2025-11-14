import tinker
import os
import asyncio
from dotenv import load_dotenv

load_dotenv()

ADAPTERS_TO_DOWNLOAD = [
    "adam_cypher_lora_v1",
    "adam_distilled_mind_v1",
    "adam_aligned_soul_v1",
    "adam_final_agent_lora" # Added the final merged adapter
]
OUTPUT_DIR = "../production_adapters"

async def download_adapter(rest_client, adapter_name):
    """Downloads a single adapter archive."""
    print(f"Downloading {adapter_name}...")
    try:
        future = await rest_client.download_checkpoint_archive_from_tinker_path_async(adapter_name)
        archive_data = await future

        output_path = os.path.join(OUTPUT_DIR, f"{adapter_name}.tar.gz")
        with open(output_path, "wb") as f:
            f.write(archive_data)

        print(f"✅ Successfully saved to {output_path}")
    except Exception as e:
        print(f"❌ Failed to download {adapter_name}: {e}")

async def main():
    """
    Asynchronously downloads the final trained adapters from Tinker cloud storage.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Initializing REST client to download adapters...")
    service_client = tinker.ServiceClient()
    rest_client = await service_client.create_rest_client_async()

    # Create a list of download tasks to run concurrently
    tasks = [download_adapter(rest_client, name) for name in ADAPTERS_TO_DOWNLOAD]
    await asyncio.gather(*tasks)

    print("\\nAdapter download complete.")

if __name__ == "__main__":
    asyncio.run(main())
