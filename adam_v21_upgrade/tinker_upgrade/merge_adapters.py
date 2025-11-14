import tinker
import os
from dotenv import load_dotenv

load_dotenv()

# --- Configuration ---
BASE_MODEL = "meta-llama/Llama-3.1-8B"
# The two adapters to be merged
MIND_ADAPTER_PATH = "adam_distilled_mind_v1"
SOUL_ADAPTER_PATH = "adam_aligned_soul_v1"
# The name for the final, combined adapter
FINAL_ADAPTER_PATH = "adam_final_agent_lora"

def merge_adapters():
    """
    Merges the 'Mind' (distilled) and 'Soul' (DPO) adapters into a single
    production-ready adapter for efficient inference.
    """
    print("--- [Final Step] Starting Adapter Merge Process ---")
    print(f"Base Model: {BASE_MODEL}")
    print(f"Merging '{MIND_ADAPTER_PATH}' (Mind) + '{SOUL_ADAPTER_PATH}' (Soul)...")

    try:
        # 1. Initialize Tinker Client
        service_client = tinker.ServiceClient()

        # 2. The `stage3_train_dpo.py` script already performed a sequential training,
        # loading the 'Mind' adapter and then training the 'Soul' on top of it.
        # The resulting `adam_aligned_soul_v1` artifact therefore already represents
        # the merged state of (Base Model + Mind + Soul).
        # This operation is primarily to rename and formally designate this
        # final artifact as the production-ready, merged adapter.
        # We use the REST client to perform this cloud storage operation.
        print(f"Renaming '{SOUL_ADAPTER_PATH}' to '{FINAL_ADAPTER_PATH}' for production deployment...")

        rest_client = service_client.create_rest_client()

        # This is a cloud storage rename/copy operation.
        rest_client.rename_checkpoint(
            path_from=SOUL_ADAPTER_PATH,
            path_to=FINAL_ADAPTER_PATH
        )

        print(f"✅ Successfully created final merged adapter: '{FINAL_ADAPTER_PATH}'")
        print("This final, composed adapter is now saved in Tinker cloud storage.")

    except Exception as e:
        print(f"❌ An error occurred during the final adapter creation process: {e}")
        print("Please ensure that both the 'Mind' and 'Soul' adapters were trained successfully.")
        exit(1)

if __name__ == "__main__":
    merge_adapters()
