import os
import json
import uuid
import glob
from typing import List, Dict, Any, Optional
from enum import Enum
from datetime import datetime
import re

# --- EMBEDDED SCRUBBER (Merged for stability) ---
class GoldStandardScrubber:
    """
    Implements the 'Gold Standard' review process:
    1. Reviews data (cleans, normalizes).
    2. Assesses conviction (scores quality).
    3. Converts to standard format (metadata extraction).
    """

    @staticmethod
    def clean_text(text: str) -> str:
        """Removes artifacts, fixes encoding, standardizes whitespace."""
        if not text:
            return ""

        # Standardize newlines
        text = text.replace('\r\n', '\n').replace('\r', '\n')

        # Remove null bytes
        text = text.replace('\x00', '')

        return text.strip()

    @staticmethod
    def assess_conviction(content: Any, artifact_type: str) -> float:
        """
        Calculates a 'Conviction Score' (0.0 - 1.0) based on quality heuristics.
        """
        score = 0.5 # Baseline

        if not content:
            return 0.0

        if isinstance(content, dict):
            # JSON Data
            keys = content.keys()
            if 'title' in keys or 'name' in keys: score += 0.1
            if 'metadata' in keys or 'meta' in keys: score += 0.1
            if len(keys) > 5: score += 0.1

            # Check for depth
            if any(isinstance(v, (dict, list)) for v in content.values()):
                score += 0.1

        elif isinstance(content, list):
            # JSONL Data
            if len(content) > 0:
                score += 0.1
                if isinstance(content[0], dict):
                    score += 0.1
            if len(content) > 10: score += 0.1

        elif isinstance(content, str):
            # Text/Markdown
            length = len(content)
            if length > 100: score += 0.1
            if length > 1000: score += 0.1

            # Structure
            if '# ' in content: score += 0.1 # Has headings
            if '```' in content: score += 0.1 # Has code blocks

        # Cap at 1.0
        return min(score, 1.0)

    @staticmethod
    def extract_metadata(content: Any, artifact_type: str) -> Dict[str, Any]:
        """Extracts useful metadata for indexing."""
        metadata = {
            "processed_at": str(datetime.now()),
            "scrubber_version": "1.0"
        }

        if isinstance(content, str):
            metadata['length'] = len(content)
            metadata['lines'] = content.count('\n') + 1
            # Extract simple entities (Capitalized words - naive)
            words = re.findall(r'\b[A-Z][a-z]+\b', content)
            if words:
                metadata['potential_entities'] = list(set(words))[:10]

        elif isinstance(content, dict):
            metadata['keys'] = list(content.keys())

        return metadata

# --- MAIN INGESTOR ---

class ArtifactType(Enum):
    REPORT = "report"
    NEWSLETTER = "newsletter"
    PROMPT = "prompt"
    DATA = "data"
    CODE_DOC = "code_doc"
    UNKNOWN = "unknown"

class GoldStandardArtifact:
    def __init__(self,
                 source_path: str,
                 content: Any,
                 artifact_type: ArtifactType,
                 title: str,
                 metadata: Dict[str, Any] = None,
                 conviction_score: float = None):
        self.id = str(uuid.uuid4())
        self.source_path = source_path
        self.content = content
        self.type = artifact_type.value
        self.title = title
        self.metadata = metadata or {}

        # Calculate conviction if not provided
        if conviction_score is None:
            self.conviction_score = GoldStandardScrubber.assess_conviction(content, self.type)
        else:
            self.conviction_score = conviction_score

        self.ingestion_timestamp = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "source_path": self.source_path,
            "type": self.type,
            "title": self.title,
            "content": self.content,
            "metadata": self.metadata,
            "conviction_score": self.conviction_score,
            "ingestion_timestamp": self.ingestion_timestamp
        }

class UniversalIngestor:
    """
    The Gold Standard Scrubbing Process.
    Ingests data from various sources, standardizes it, and assesses conviction.
    """
    def __init__(self):
        self.artifacts: List[GoldStandardArtifact] = []

    def scan_directory(self, root_path: str, recursive: bool = True):
        """Scans a directory for ingestible content."""
        if not os.path.exists(root_path):
            print(f"Warning: Path {root_path} does not exist.")
            return

        print(f"Scanning {root_path}...")

        # Walk through directory
        for root, dirs, files in os.walk(root_path):
            # Skip hidden dirs
            dirs[:] = [d for d in dirs if not d.startswith('.')]

            for file in files:
                filepath = os.path.join(root, file)
                self.process_file(filepath)
            if not recursive:
                break

    def process_file(self, filepath: str):
        """Determines file type and processes it."""
        filename = os.path.basename(filepath)

        # Skip hidden files and python files (unless we want to document code)
        if filename.startswith('.') or filename.startswith('__'):
            return

        # Skip this file itself and generated files to avoid loops
        if "gold_standard" in filepath or "ui_data.json" in filepath:
            return

        try:
            if filepath.endswith('.json'):
                self._process_json(filepath)
            elif filepath.endswith('.jsonl'):
                self._process_jsonl(filepath)
            elif filepath.endswith('.md'):
                self._process_markdown(filepath)
            elif filepath.endswith('.txt'):
                self._process_text(filepath)
        except Exception as e:
            # print(f"Error processing {filepath}: {e}")
            pass

    def _process_json(self, filepath: str):
        with open(filepath, 'r', encoding='utf-8') as f:
            try:
                content = f.read()
                clean_content = GoldStandardScrubber.clean_text(content)
                data = json.loads(clean_content)
            except json.JSONDecodeError:
                return # Skip invalid JSON

        # Determine type based on path or content
        artifact_type = ArtifactType.DATA
        title = os.path.basename(filepath)

        if "reports" in filepath:
            artifact_type = ArtifactType.REPORT
            title = data.get("title", data.get("company_name", title))
        elif "prompt" in filepath:
            artifact_type = ArtifactType.PROMPT

        # Extract metadata
        metadata = GoldStandardScrubber.extract_metadata(data, artifact_type.value)
        if "original_keys" not in metadata:
             metadata["original_keys"] = list(data.keys()) if isinstance(data, dict) else []

        # Create artifact
        artifact = GoldStandardArtifact(
            source_path=filepath,
            content=data,
            artifact_type=artifact_type,
            title=title,
            metadata=metadata
        )
        self.artifacts.append(artifact)

    def _process_jsonl(self, filepath: str):
        # Treat the whole file as a dataset artifact
        content = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    content.append(json.loads(line))
                except:
                    continue

        title = os.path.basename(filepath)
        artifact = GoldStandardArtifact(
            source_path=filepath,
            content=content,
            artifact_type=ArtifactType.DATA,
            title=title,
            metadata={"record_count": len(content)}
        )
        self.artifacts.append(artifact)

    def _process_markdown(self, filepath: str):
        with open(filepath, 'r', encoding='utf-8') as f:
            text = GoldStandardScrubber.clean_text(f.read())

        # Extract title from first line if possible
        lines = text.split('\n')
        title = os.path.basename(filepath)
        for line in lines:
            if line.strip().startswith('# '):
                title = line.strip().replace('# ', '')
                break

        artifact_type = ArtifactType.CODE_DOC
        if "prompt" in filepath:
            artifact_type = ArtifactType.PROMPT
        elif "newsletter" in filepath or "Fortress" in filepath:
            artifact_type = ArtifactType.NEWSLETTER

        metadata = GoldStandardScrubber.extract_metadata(text, artifact_type.value)

        artifact = GoldStandardArtifact(
            source_path=filepath,
            content=text,
            artifact_type=artifact_type,
            title=title,
            metadata=metadata
        )
        self.artifacts.append(artifact)

    def _process_text(self, filepath: str):
        with open(filepath, 'r', encoding='utf-8') as f:
            text = GoldStandardScrubber.clean_text(f.read())

        metadata = GoldStandardScrubber.extract_metadata(text, ArtifactType.UNKNOWN.value)

        artifact = GoldStandardArtifact(
            source_path=filepath,
            content=text,
            artifact_type=ArtifactType.UNKNOWN,
            title=os.path.basename(filepath),
            metadata=metadata
        )
        self.artifacts.append(artifact)

    def save_to_jsonl(self, output_path: str):
        """Saves all artifacts to a JSONL file."""
        with open(output_path, 'w', encoding='utf-8') as f:
            for artifact in self.artifacts:
                f.write(json.dumps(artifact.to_dict()) + '\n')
        print(f"Saved {len(self.artifacts)} artifacts to {output_path}")

    def get_artifacts_by_type(self, artifact_type: ArtifactType) -> List[Dict]:
        return [a.to_dict() for a in self.artifacts if a.type == artifact_type.value]

if __name__ == "__main__":
    # Example usage
    ingestor = UniversalIngestor()
    ingestor.scan_directory("core/libraries_and_archives")
    ingestor.scan_directory("prompt_library")
    ingestor.scan_directory("data")

    # Also scan docs for newsletters/manuals
    ingestor.scan_directory("docs")

    # Create output directory if it doesn't exist
    os.makedirs("data/gold_standard", exist_ok=True)
    ingestor.save_to_jsonl("data/gold_standard/knowledge_artifacts.jsonl")
