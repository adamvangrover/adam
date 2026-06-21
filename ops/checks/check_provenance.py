import sys
import json
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_check():
    logging.info("Checking provenance header validation in daily data...")
    try:
        # Check daily dataset
        with open('showcase/data/adam_daily/2026-06-04/data.jsonl', 'r') as f:
            # Skip the first metadata line
            next(f)
            for line in f:
                data = json.loads(line)
                assert 'context_provenance' in data

        logging.info("Provenance check completed successfully.")
        sys.exit(0)
    except Exception as e:
        logging.error(f"Error during provenance check: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_check()
