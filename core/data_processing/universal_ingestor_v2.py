import os
import json
import uuid
import hashlib
import ast
import re
import logging
import argparse
from typing import List, Dict, Any, Optional, Callable, Union
from enum import Enum
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict

# --- CONFIGURATION & LOGGING ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("UniversalIngestor")

# --- DATA STRUCTURES ---

class ArtifactType(Enum):
    REPORT = "report"
    NEWSLETTER = "newsletter"
    PROMPT = "prompt"
    DATA = "data"
    CODE_DOC = "code_doc"
    CONFIG = "config"
    UNKNOWN = "unknown"

@dataclass
class GoldStandardArtifact:
    """Standardized Data Object for all knowledge assets."""
    source_path: str
    content: Any
    artifact_type: str
    title: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    conviction_score: float = 0.5
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content_hash: str = ""
    ingestion_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

# --- SCORING & CLEANING ENGINE ---

class GoldStandardScrubber:
    """
    Stateless engine for cleaning text and calculating conviction.
    """
    
    # Configurable weights for scoring
    WEIGHTS = {
        'baseline': 0.3,
        'metadata_richness': 0.1,
        'structural_depth': 0.15,
        'code_documentation': 0.2,
        'text_length_bonus': 0.1,
        'formatting_bonus': 0.15
    }

    @staticmethod
    def compute_file_hash(content: Union[str, bytes]) -> str:
        """Generates MD5 hash for change detection."""
        if isinstance(content, str):
            content = content.encode('utf-8')
        return hashlib.md5(content).hexdigest()

    @staticmethod
    def clean_text(text: str) -> str:
        if not text: return ""
        # Normalize line endings and strip nulls
        text = text.replace('\r\n', '\n').replace('\r', '\n').replace('\x00', '')
        # Remove excessive whitespace while preserving paragraph structure
        return re.sub(r'\n{3,}', '\n\n', text).strip()

    @classmethod
    def assess_conviction(cls, content: Any, artifact_type: str) -> float:
        score = cls.WEIGHTS['baseline']

        # 1. Dictionary/JSON Heuristics
        if isinstance(content, dict):
            keys = content.keys()
            if len(keys) > 4: score += cls.WEIGHTS['metadata_richness']
            if 'metadata' in keys or 'meta' in keys: score += 0.1
            # Check for nesting depth (complexity)
            if any(isinstance(v, (dict, list)) for v in content.values()):
                score += cls.WEIGHTS['structural_depth']

        # 2. List/JSONL Heuristics
        elif isinstance(content, list):
            if len(content) > 5: score += 0.1
            if len(content) > 0 and isinstance(content[0], dict):
                score += cls.WEIGHTS['structural_depth']

        # 3. Text/Markdown/Code Heuristics
        elif isinstance(content, str):
            length = len(content)
            if length > 500: score += cls.WEIGHTS['text_length_bonus']
            
            # Markdown Structure
            if '# ' in content or '## ' in content: 
                score += cls.WEIGHTS['formatting_bonus']
            
            # Code Documentation specific
            if artifact_type == ArtifactType.CODE_DOC.value:
                if "Args:" in content or "Returns:" in content: score += 0.2
                if "class " in content: score += 0.1

        return min(round(score, 2), 1.0)

    @staticmethod
    def extract_metadata(content: Any) -> Dict[str, Any]:
        """Extracts statistical metadata."""
        meta = {"processed_at": datetime.now().isoformat()}
        
        if isinstance(content, str):
            meta.update({
                "char_count": len(content),
                "line_count": content.count('\n') + 1,
                # Simple extraction of capitalized terms (potential entities)
                "tags": list(set(re.findall(r'#\w+', content)))[:5],
                "potential_entities": list(set(re.findall(r'\b[A-Z][a-z]{3,}\b', content)))[:8]
            })
        elif isinstance(content, dict):
            meta["top_level_keys"] = list(content.keys())
        
        return meta

# --- FILE HANDLERS (STRATEGY PATTERN) ---

class FileHandlers:
    """Encapsulates logic for parsing specific file types."""
    
    @staticmethod
    def handle_json(filepath: str, raw_text: str) -> Optional[GoldStandardArtifact]:
        try:
            data = json.loads(raw_text)
        except json.JSONDecodeError:
            logger.warning(f"Invalid JSON in {filepath}")
            return None

        # Determine type
        a_type = ArtifactType.DATA
        if "reports" in filepath or "report" in data.get("type", ""):
            a_type = ArtifactType.REPORT
        elif "prompt" in filepath:
            a_type = ArtifactType.PROMPT

        title = data.get("title", data.get("name", os.path.basename(filepath)))
        
        return GoldStandardArtifact(
            source_path=filepath,
            content=data,
            artifact_type=a_type.value,
            title=title,
            metadata=GoldStandardScrubber.extract_metadata(data),
            content_hash=GoldStandardScrubber.compute_file_hash(raw_text)
        )

    @staticmethod
    def handle_jsonl(filepath: str, raw_text: str) -> Optional[GoldStandardArtifact]:
        data = []
        for line in raw_text.splitlines():
            try:
                if line.strip(): data.append(json.loads(line))
            except: continue
            
        if not data: return None

        return GoldStandardArtifact(
            source_path=filepath,
            content=data,
            artifact_type=ArtifactType.DATA.value,
            title=os.path.basename(filepath),
            metadata={"record_count": len(data)},
            content_hash=GoldStandardScrubber.compute_file_hash(raw_text)
        )

    @staticmethod
    def handle_markdown(filepath: str, raw_text: str) -> Optional[GoldStandardArtifact]:
        clean_text = GoldStandardScrubber.clean_text(raw_text)
        
        # Extract title from first H1 header
        title = os.path.basename(filepath)
        match = re.search(r'^#\s+(.+)$', clean_text, re.MULTILINE)
        if match:
            title = match.group(1).strip()

        a_type = ArtifactType.CODE_DOC
        if "prompt" in filepath.lower(): a_type = ArtifactType.PROMPT
        elif "newsletter" in filepath.lower(): a_type = ArtifactType.NEWSLETTER

        return GoldStandardArtifact(
            source_path=filepath,
            content=clean_text,
            artifact_type=a_type.value,
            title=title,
            metadata=GoldStandardScrubber.extract_metadata(clean_text),
            content_hash=GoldStandardScrubber.compute_file_hash(raw_text)
        )

    @staticmethod
    def handle_python(filepath: str, raw_text: str) -> Optional[GoldStandardArtifact]:
        try:
            tree = ast.parse(raw_text)
            docstring = ast.get_docstring(tree)
            
            # Extract Classes and Functions
            classes = [n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
            functions = [n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
            
            # Cyclomatic Complexity heuristic (counting decision points)
            complexity = raw_text.count('if ') + raw_text.count('for ') + raw_text.count('while ')

            summary = f"Module: {os.path.basename(filepath)}\n"
            if docstring: summary += f"Docstring: {docstring}\n"
            summary += f"Classes: {', '.join(classes)}\nFunctions: {', '.join(functions)}"

            meta = {
                "classes": classes, 
                "functions": functions, 
                "complexity_score": complexity
            }

            return GoldStandardArtifact(
                source_path=filepath,
                content=summary,
                artifact_type=ArtifactType.CODE_DOC.value,
                title=f"Code: {os.path.basename(filepath)}",
                metadata=meta,
                content_hash=GoldStandardScrubber.compute_file_hash(raw_text)
            )
        except Exception as e:
            logger.error(f"AST Parse Error {filepath}: {e}")
            return None

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
            '.py': FileHandlers.handle_python
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
