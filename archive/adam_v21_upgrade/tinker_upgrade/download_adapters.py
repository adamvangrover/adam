import tinker
import os
from dotenv import load_dotenv

load_dotenv()

ADAPTERS_TO_DOWNLOAD = [
    "adam_cypher_lora_v1",
    "adam_distilled_mind_v1",
    "adam_aligned_soul_v1"
]
OUTPUT_DIR = "../production_adapters"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Initializing REST client to download adapters...")
service_client = tinker.ServiceClient()
rest_client = service_client.create_rest_client()

for adapter_name in ADAPTERS_TO_DOWNLOAD:
    print(f"Downloading {adapter_name}...")
    try:
        future = rest_client.download_checkpoint_archive_from_tinker_path(adapter_name)
        output_path = os.path.join(OUTPUT_DIR, f"{adapter_name}.tar.gz")
        # .result() blocks until download is complete
        with open(output_path, "wb") as f:
            f.write(future.result())
        print(f"✅ Successfully saved to {output_path}")
    except Exception as e:
        print(f"❌ Failed to download {adapter_name}: {e}")

print("Adapter download complete.")
