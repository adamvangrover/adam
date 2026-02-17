"""
Batch processing script to run the Credit Memo RAG pipeline for multiple tickers.
"""
import os
import sys
import subprocess
import time

# List of tickers to process
TICKERS = ["MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "JPM", "GS", "NFLX"]

def main():
    print(f"Starting batch RAG processing for {len(TICKERS)} tickers...")

    for ticker in TICKERS:
        file_path = f"data/10k_sample_{ticker.lower()}.txt"

        if not os.path.exists(file_path):
            print(f"SKIPPING {ticker}: File {file_path} not found.")
            continue

        print(f"Processing {ticker} using {file_path}...")

        # Construct command
        cmd = [sys.executable, "scripts/run_credit_memo_rag.py", "--ticker", ticker, "--file", file_path]

        # Run process
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(f"SUCCESS {ticker}")
            # print(result.stdout) # Optional verbose output
        except subprocess.CalledProcessError as e:
            print(f"FAILURE {ticker}: {e}")
            print(e.stderr)

        # Small delay to prevent race conditions on file writes if any
        time.sleep(0.5)

    print("Batch processing complete.")

if __name__ == "__main__":
    main()
