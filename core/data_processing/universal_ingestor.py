import os
import json
import uuid
import glob
from typing import List, Dict, Any, Optional
from enum import Enum
from datetime import datetime

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
                 conviction_score: float = 0.95):
        self.id = str(uuid.uuid4())
        self.source_path = source_path
        self.content = content
        self.type = artifact_type.value
        self.title = title
        self.metadata = metadata or {}
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
                data = json.load(f)
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

        # Create artifact
        artifact = GoldStandardArtifact(
            source_path=filepath,
            content=data,
            artifact_type=artifact_type,
            title=title,
            metadata={"original_keys": list(data.keys()) if isinstance(data, dict) else []}
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
            text = f.read()

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

        artifact = GoldStandardArtifact(
            source_path=filepath,
            content=text,
            artifact_type=artifact_type,
            title=title
        )
        self.artifacts.append(artifact)

    def _process_text(self, filepath: str):
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()

        artifact = GoldStandardArtifact(
            source_path=filepath,
            content=text,
            artifact_type=ArtifactType.UNKNOWN,
            title=os.path.basename(filepath)
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

    # Create output directory if it doesn't exist
    os.makedirs("data/gold_standard", exist_ok=True)
    ingestor.save_to_jsonl("data/gold_standard/knowledge_artifacts.jsonl")
