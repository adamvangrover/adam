import os
import sys
import argparse
import logging
from pathlib import Path

# Ensure root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.utils.repo_compiler.scanner import RepoScanner
from core.utils.repo_compiler.chunker import RepoChunker
from core.utils.repo_compiler.formatter import PromptFormatter
from core.utils.repo_compiler.summarizer import ChunkSummarizer

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Compile the repository into an LLM prompt.")
    parser.add_argument("--root", default=".", help="Root directory of the repository.")
    parser.add_argument("--output", default="repo_prompt.txt", help="Output file path.")
    parser.add_argument("--mode", choices=["monolith", "chunk_size", "chunk_dir"], default="monolith",
                        help="How to group the output.")
    parser.add_argument("--format", choices=["markdown", "xml"], default="markdown",
                        help="Format of the prompt wrapping.")
    parser.add_argument("--max-size-mb", type=float, default=1.0,
                        help="Skip files larger than this size in MB.")
    parser.add_argument("--max-tokens", type=int, default=100000,
                        help="Max tokens per chunk (used with chunk_size mode).")
    parser.add_argument("--summarize", action="store_true",
                        help="Use litellm to generate summaries for chunks.")
    parser.add_argument("--system-prompt", default="You are an expert AI software engineer analyzing this codebase.",
                        help="Optional system prompt to prepend.")

    args = parser.parse_args()

    logger.info(f"Scanning repository at {args.root}...")
    scanner = RepoScanner(root_dir=args.root, max_file_size_mb=args.max_size_mb)
    documents = scanner.scan()
    logger.info(f"Found {len(documents)} valid text files.")

    formatter = PromptFormatter()

    if args.mode == "monolith":
        logger.info(f"Formatting as a single monolith document...")
        output_text = formatter.format_monolith(documents, format_type=args.format, system_prompt=args.system_prompt)

        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output_text)
        logger.info(f"Wrote monolithic prompt to {args.output}")

    else:
        chunker = RepoChunker()
        if args.mode == "chunk_size":
            logger.info(f"Chunking by token limit ({args.max_tokens})...")
            chunks = chunker.chunk_by_token_limit(documents, max_tokens=args.max_tokens)
        else: # chunk_dir
            logger.info(f"Chunking by root directory...")
            chunks = chunker.chunk_by_directory(documents)

        logger.info(f"Created {len(chunks)} chunks.")

        summarizer = None
        if args.summarize:
            summarizer = ChunkSummarizer()

        output_dir = Path(args.output).parent
        if output_dir.name and not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)

        base_name = Path(args.output).stem
        ext = Path(args.output).suffix or ".txt"

        for i, chunk in enumerate(chunks):
            if summarizer:
                logger.info(f"Summarizing chunk {chunk.chunk_id}...")
                chunk = summarizer.summarize(chunk)

            chunk_text = formatter.format_chunk(chunk, format_type=args.format)

            chunk_filename = output_dir / f"{base_name}_{chunk.chunk_id}{ext}"
            with open(chunk_filename, "w", encoding="utf-8") as f:
                if args.system_prompt:
                    f.write(args.system_prompt + "\n\n")
                f.write(chunk_text)

            logger.info(f"Wrote chunk to {chunk_filename}")

if __name__ == "__main__":
    main()
