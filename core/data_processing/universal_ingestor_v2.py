import os
import json
import logging
import argparse
import hashlib
from typing import List, Dict, Callable, Optional
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

# Import common utilities
from core.data_processing.utils import (
    GoldStandardArtifact,
    GoldStandardScrubber,
    FileHandlers,
    ArtifactType
)

# --- CONFIGURATION & LOGGING ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("UniversalIngestor")

# --- MAIN SYSTEM ---

class UniversalIngestor:
    def __init__(self, state_file: str = "ingestion_state.json"):
        self.artifacts: List[GoldStandardArtifact] = []
        self.state_file = state_file
        self.previous_state = self._load_state()
        self.current_state = {}
        
        # Strategy Mapping
        self.handlers: Dict[str, Callable] = {
            '.json': FileHandlers.handle_json,
            '.jsonl': FileHandlers.handle_jsonl,
            '.md': FileHandlers.handle_markdown,
            '.txt': FileHandlers.handle_markdown, # Treat txt as MD for now
            '.py': FileHandlers.handle_python,
            '.parquet': FileHandlers.handle_parquet
        }

    def _load_state(self) -> Dict[str, str]:
        """Loads the hash map of the previous run."""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as f:
                    return json.load(f)
            except: pass
        return {}

    def _process_single_file(self, filepath: str) -> Optional[GoldStandardArtifact]:
        """Worker function for parallel processing."""
        ext = os.path.splitext(filepath)[1].lower()
        if ext not in self.handlers:
            return None

        try:
            if ext == '.parquet':
                raw_text = None
                current_hash = hashlib.md5(filepath.encode()).hexdigest()
            else:
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    raw_text = f.read()

                # Check Hash (Incremental Ingestion)
                current_hash = GoldStandardScrubber.compute_file_hash(raw_text)
            
            # You would return a special "Unchanged" signal here in a real DB sync, 
            # but for a static rebuild, we just update the hash map and process.
            # (To truly skip, we'd need to load the old artifact from storage, 
            # but here we are regenerating the JSONL, so we must re-process or load from cache).
            
            # For this implementation, we re-process to ensure JSONL is complete,
            # but we use the hash to potentially flag 'updated' vs 'new'.
            
            artifact = self.handlers[ext](filepath, raw_text)
            
            if artifact:
                # Late-binding conviction score
                artifact.conviction_score = GoldStandardScrubber.assess_conviction(
                    artifact.content, artifact.artifact_type
                )
                return artifact
                
        except Exception as e:
            logger.error(f"Failed to process {filepath}: {e}")
            return None
        
        return None

    def scan_and_process(self, root_paths: List[str], max_workers: int = 4):
        """
        Main entry point. Uses ProcessPoolExecutor for parallel ingestion.
        """
        all_files = []
        for root_path in root_paths:
            for root, dirs, files in os.walk(root_path):
                dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
                for file in files:
                    if not file.startswith('.'):
                        all_files.append(os.path.join(root, file))

        logger.info(f"Found {len(all_files)} files. Starting parallel ingestion...")

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # map returns results in order, as_completed returns as they finish
            future_to_file = {executor.submit(self._process_single_file, fp): fp for fp in all_files}
            
            for future in as_completed(future_to_file):
                result = future.result()
                if result:
                    self.artifacts.append(result)
                    self.current_state[result.source_path] = result.content_hash

    def save_output(self, output_path: str):
        # Sort artifacts by type for cleaner viewing in the file
        self.artifacts.sort(key=lambda x: x.artifact_type)
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for art in self.artifacts:
                f.write(json.dumps(art.to_dict()) + '\n')
        
        # Save state
        with open(self.state_file, 'w', encoding='utf-8') as f:
            json.dump(self.current_state, f, indent=2)
            
        logger.info(f"Ingestion complete. {len(self.artifacts)} artifacts saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Adam v23.0 Universal Ingestor")
    parser.add_argument("--dirs", nargs='+', default=["core", "data", "docs", "prompt_library"], help="Directories to scan")
    parser.add_argument("--output", default="data/gold_standard/knowledge_artifacts.jsonl", help="Output file path")
    parser.add_argument("--workers", type=int, default=os.cpu_count(), help="Number of parallel workers")
    
    args = parser.parse_args()
    
    ingestor = UniversalIngestor()
    ingestor.scan_and_process(args.dirs, max_workers=args.workers)
    ingestor.save_output(args.output)
