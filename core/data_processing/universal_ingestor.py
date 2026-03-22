import ast
import asyncio
import json
import re
import uuid
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import aiofiles

from core.data_processing.chunking_engine import ChunkingEngine

# --- EMBEDDED SCRUBBER (Merged for stability) ---


class GoldStandardScrubber:
    """
    Implements the 'Gold Standard' review process:
    1. Reviews data (cleans, normalizes).
    2. Assesses conviction (scores quality).
    3. Converts to standard format (metadata extraction).
    """

    # Bolt Optimization: Pre-compile regex for performance
    ENTITY_REGEX = re.compile(r'\b[A-Z][a-z]+\b')

    # Bolt Optimization: Move keywords to class constant
    HIGH_IMPACT_KEYWORDS = (
        "confidential", "proprietary", "strategy", "roadmap",
        "quarterly report", "financial statement", "10-k", "10-q",
        "risk assessment", "audit", "compliance"
    )

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
        score = 0.5  # Baseline

        if not content:
            return 0.0

        if isinstance(content, dict):
            # JSON Data
            keys = content.keys()
            if 'title' in keys or 'name' in keys:
                score += 0.1
            if 'metadata' in keys or 'meta' in keys:
                score += 0.1
            if len(keys) > 5:
                score += 0.1

            # Check for depth
            if any(isinstance(v, (dict, list)) for v in content.values()):
                score += 0.1

        elif isinstance(content, list):
            # JSONL Data
            if len(content) > 0:
                score += 0.1
                if isinstance(content[0], dict):
                    score += 0.1
            if len(content) > 10:
                score += 0.1

        elif isinstance(content, str):
            # Text/Markdown
            length = len(content)
            if length > 100:
                score += 0.1
            if length > 1000:
                score += 0.1

            # Structure
            if '# ' in content:
                score += 0.1  # Has headings
            if '```' in content:
                score += 0.1  # Has code blocks

            # Python Docstrings
            if artifact_type == "code_doc":
                if "Args:" in content or "Returns:" in content:
                    score += 0.2
                if "class " in content:
                    score += 0.1

        # Cap at 1.0
        return min(score, 1.0)

    @staticmethod
    def is_high_impact(content: str, artifact_type: str) -> bool:
        """Determines if the content is 'high impact' and warrants a deeper scan."""
        # Check first 2KB for keywords
        head = content[:2048].lower()

        if any(kw in head for kw in GoldStandardScrubber.HIGH_IMPACT_KEYWORDS):
            return True

        if artifact_type in ["report", "newsletter"]:
            return True

        return False

    @staticmethod
    def extract_metadata(content: Any, artifact_type: str) -> dict[str, Any]:
        """Extracts useful metadata for indexing using an adaptive scan strategy."""
        metadata = {
            "processed_at": str(datetime.now()),
            "scrubber_version": "1.2",
            "scan_strategy": "QUICK"
        }

        if isinstance(content, str):
            length = len(content)
            metadata['length'] = length
            metadata['lines'] = content.count('\n') + 1

            # Adaptive Scan Strategy
            scan_limit = 10000  # Default QUICK scan

            if GoldStandardScrubber.is_high_impact(content, artifact_type):
                scan_limit = 100000  # DEEP scan for high impact docs
                metadata['scan_strategy'] = "DEEP"

            # Flag massive files for human review
            if length > 5 * 1024 * 1024:  # > 5MB
                metadata['review_required'] = True
                metadata['review_reason'] = "LARGE_FILE_SIZE"

            # Extract entities with adaptive limit using pre-compiled regex
            words = GoldStandardScrubber.ENTITY_REGEX.findall(content[:scan_limit])
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
    """
    Architecture & Usage:
    A standardized container for data artifacts ingested by the system.
    This class enforces a consistent schema across various data modalities (JSON, Markdown, Python)
    ensuring downstream pipelines (e.g., Vector DBs, LangGraph models) have predictable inputs.

    Attributes:
        id (str): A unique UUID for the artifact.
        source_path (str): The original file path.
        content (Any): The parsed data payload.
        type (str): The enum string representation of the ArtifactType.
        title (str): The inferred title.
        metadata (dict): Extracted metadata.
        conviction_score (float): A heuristic-based quality score (0.0 - 1.0).
        ingestion_timestamp (str): ISO 8601 formatted timestamp of processing.
    """
    def __init__(self,
                 source_path: str,
                 content: Any,
                 artifact_type: ArtifactType,
                 title: str,
                 metadata: dict[str, Any] | None = None,
                 conviction_score: float | None = None):
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

    def to_dict(self) -> dict[str, Any]:
        """Serializes the artifact into a JSON-compatible dictionary format."""
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
    Architecture & Usage:
    The Universal Ingestor handles the 'Gold Standard' data processing pipeline.
    It recurses through file directories, delegates to the appropriate parser based on file extension,
    extracts meaningful metadata and heuristic conviction scores, and serializes the
    normalized outputs into a standardized JSONL format for the knowledge base.
    """

    def __init__(self):
        self.artifacts: list[GoldStandardArtifact] = []
        self.chunking_engine = ChunkingEngine(strategy="semantic")

    def scan_directory(self, root_path: str, recursive: bool = True):
        """
        Walks a directory and routes all permitted files to the processing pipeline.

        Args:
            root_path (str): The starting directory path.
            recursive (bool): Whether to traverse subdirectories. Defaults to True.
        """
        root = Path(root_path)
        if not root.exists():
            print(f"Warning: Path {root_path} does not exist.")
            return

        print(f"Scanning {root_path}...")

        pattern = "**/*" if recursive else "*"
        for filepath in root.glob(pattern):
            if filepath.is_file():
                if any(part.startswith('.') for part in filepath.parts):
                    continue
                self.process_file(str(filepath))

    def process_file(self, filepath: str):
        """
        Determines file type, enforces safety constraints (e.g., file size limits),
        and routes the path to the correct internal processor.

        Args:
            filepath (str): The absolute or relative path to the file.
        """
        path = Path(filepath)
        filename = path.name

        if filename.startswith('.') or filename == "__init__.py":
            return

        # Skip this file itself and generated files to avoid loops
        if "gold_standard" in filepath or "ui_data.json" in filepath:
            return

        # ROBUSTNESS: Safe Mode Check
        try:
            if path.stat().st_size > 10 * 1024 * 1024:
                print(f"Skipping large file: {filepath} (>10MB)")
                return
        except OSError:
            return

        ext = path.suffix.lower()
        processor_map = {
            '.json': self._process_json,
            '.jsonl': self._process_jsonl,
            '.md': self._process_markdown,
            '.txt': self._process_text,
            '.log': self._process_text,
            '.py': self._process_python,
        }

        processor = processor_map.get(ext)
        if processor:
            try:
                processor(path)
            except Exception:
                pass

    async def scan_directory_async(self, root_path: str, recursive: bool = True, max_concurrent: int = 50):
        """
        Asynchronously walks a directory and routes all permitted files to the processing pipeline.
        Uses a semaphore to limit concurrent file descriptors.
        """
        root = Path(root_path)
        if not root.exists():
            print(f"Warning: Path {root_path} does not exist.")
            return

        print(f"Async Scanning {root_path}...")

        semaphore = asyncio.Semaphore(max_concurrent)

        pattern = "**/*" if recursive else "*"
        tasks = []
        for filepath in root.glob(pattern):
            if filepath.is_file():
                if any(part.startswith('.') for part in filepath.parts):
                    continue
                tasks.append(self._process_with_semaphore(semaphore, str(filepath)))

        if tasks:
            await asyncio.gather(*tasks)

    async def _process_with_semaphore(self, semaphore: asyncio.Semaphore, filepath: str):
        async with semaphore:
            await self.process_file_async(filepath)

    async def process_file_async(self, filepath: str):
        """
        Asynchronously routes the path to the correct internal processor.
        """
        path = Path(filepath)
        filename = path.name

        if filename.startswith('.') or filename == "__init__.py":
            return

        if "gold_standard" in filepath or "ui_data.json" in filepath:
            return

        try:
            if path.stat().st_size > 10 * 1024 * 1024:
                print(f"Skipping large file: {filepath} (>10MB)")
                return
        except OSError:
            return

        ext = path.suffix.lower()
        processor_map = {
            '.json': self._process_json_async,
            '.jsonl': self._process_jsonl_async,
            '.md': self._process_markdown_async,
            '.txt': self._process_text_async,
            '.log': self._process_text_async,
            '.py': self._process_python_async,
        }

        processor = processor_map.get(ext)
        if processor:
            try:
                await processor(path)
            except Exception:
                pass

    def _process_json(self, filepath: Path):
        with filepath.open('r', encoding='utf-8', errors='ignore') as f:
            try:
                content = f.read()
                clean_content = GoldStandardScrubber.clean_text(content)
                data = json.loads(clean_content)
            except json.JSONDecodeError:
                return  # Skip invalid JSON

        artifact_type = ArtifactType.DATA
        title = filepath.name

        if "reports" in str(filepath):
            artifact_type = ArtifactType.REPORT
            title = data.get("title", data.get("company_name", title))
        elif "prompt" in str(filepath):
            artifact_type = ArtifactType.PROMPT

        metadata = GoldStandardScrubber.extract_metadata(data, artifact_type.value)
        if "original_keys" not in metadata:
            metadata["original_keys"] = list(data.keys()) if isinstance(data, dict) else []

        artifact = GoldStandardArtifact(
            source_path=str(filepath),
            content=data,
            artifact_type=artifact_type,
            title=title,
            metadata=metadata
        )
        self.artifacts.append(artifact)

    def _process_jsonl(self, filepath: Path):
        content = []
        with filepath.open('r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                try:
                    content.append(json.loads(line))
                except Exception:
                    continue

        title = filepath.name
        artifact = GoldStandardArtifact(
            source_path=str(filepath),
            content=content,
            artifact_type=ArtifactType.DATA,
            title=title,
            metadata={"record_count": len(content)}
        )
        self.artifacts.append(artifact)

    def _process_markdown(self, filepath: Path):
        with filepath.open('r', encoding='utf-8', errors='ignore') as f:
            text = GoldStandardScrubber.clean_text(f.read())

        lines = text.split('\n')
        title = filepath.name

        for line in lines:
            if line.strip().startswith('# '):
                title = line.strip().replace('# ', '')
                break

        # Semantic Chunking via ChunkingEngine
        chunk_dicts = self.chunking_engine.chunk(text)
        chunks = [c["text"] for c in chunk_dicts if "text" in c]

        artifact_type = ArtifactType.CODE_DOC
        if "prompt" in str(filepath):
            artifact_type = ArtifactType.PROMPT
        elif "newsletter" in str(filepath) or "Fortress" in str(filepath):
            artifact_type = ArtifactType.NEWSLETTER

        metadata = GoldStandardScrubber.extract_metadata(text, artifact_type.value)

        # Vector Database / RAG embedding prep
        metadata["semantic_chunks"] = len(chunks)
        metadata["chunk_preview"] = chunks[0][:100] + "..." if chunks else ""

        artifact = GoldStandardArtifact(
            source_path=str(filepath),
            content={"full_text": text, "chunks": chunks},
            artifact_type=artifact_type,
            title=title,
            metadata=metadata
        )
        self.artifacts.append(artifact)

    def _process_text(self, filepath: Path):
        with filepath.open('r', encoding='utf-8', errors='ignore') as f:
            text = GoldStandardScrubber.clean_text(f.read())

        # Semantic Chunking via ChunkingEngine
        chunk_dicts = self.chunking_engine.chunk(text)
        chunks = [c["text"] for c in chunk_dicts if "text" in c]

        metadata = GoldStandardScrubber.extract_metadata(text, ArtifactType.UNKNOWN.value)
        metadata["semantic_chunks"] = len(chunks)
        metadata["chunk_preview"] = chunks[0][:100] + "..." if chunks else ""

        artifact = GoldStandardArtifact(
            source_path=str(filepath),
            content={"full_text": text, "chunks": chunks},
            artifact_type=ArtifactType.UNKNOWN,
            title=filepath.name,
            metadata=metadata
        )
        self.artifacts.append(artifact)

    def _process_python(self, filepath: Path):
        try:
            with filepath.open('r', encoding='utf-8', errors='ignore') as f:
                source = f.read()

            tree = ast.parse(source)
            docstring = ast.get_docstring(tree)

            classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
            functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]

            content = f"Module: {filepath.name}\n"
            if docstring:
                content += f"Docstring: {docstring}\n"
            if classes:
                content += f"Classes: {', '.join(classes)}\n"
            if functions:
                content += f"Functions: {', '.join(functions)}\n"

            metadata = {
                "classes": classes,
                "functions": functions,
                "has_docstring": bool(docstring)
            }

            artifact = GoldStandardArtifact(
                source_path=str(filepath),
                content=content,
                artifact_type=ArtifactType.CODE_DOC,
                title=f"Doc: {filepath.name}",
                metadata=metadata
            )
            self.artifacts.append(artifact)

        except Exception:
            pass

    async def _process_json_async(self, filepath: Path):
        async with aiofiles.open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            try:
                content = await f.read()
                clean_content = GoldStandardScrubber.clean_text(content)
                data = json.loads(clean_content)
            except json.JSONDecodeError:
                return  # Skip invalid JSON

        artifact_type = ArtifactType.DATA
        title = filepath.name

        if "reports" in str(filepath):
            artifact_type = ArtifactType.REPORT
            title = data.get("title", data.get("company_name", title))
        elif "prompt" in str(filepath):
            artifact_type = ArtifactType.PROMPT

        metadata = GoldStandardScrubber.extract_metadata(data, artifact_type.value)
        if "original_keys" not in metadata:
            metadata["original_keys"] = list(data.keys()) if isinstance(data, dict) else []

        artifact = GoldStandardArtifact(
            source_path=str(filepath),
            content=data,
            artifact_type=artifact_type,
            title=title,
            metadata=metadata
        )
        self.artifacts.append(artifact)

    async def _process_jsonl_async(self, filepath: Path):
        content = []
        async with aiofiles.open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            async for line in f:
                try:
                    content.append(json.loads(line))
                except Exception:
                    continue

        title = filepath.name
        artifact = GoldStandardArtifact(
            source_path=str(filepath),
            content=content,
            artifact_type=ArtifactType.DATA,
            title=title,
            metadata={"record_count": len(content)}
        )
        self.artifacts.append(artifact)

    async def _process_markdown_async(self, filepath: Path):
        async with aiofiles.open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            raw_text = await f.read()
            text = GoldStandardScrubber.clean_text(raw_text)

        lines = text.split('\n')
        title = filepath.name

        for line in lines:
            if line.strip().startswith('# '):
                title = line.strip().replace('# ', '')
                break

        # Semantic Chunking via ChunkingEngine
        chunk_dicts = self.chunking_engine.chunk(text)
        chunks = [c["text"] for c in chunk_dicts if "text" in c]

        artifact_type = ArtifactType.CODE_DOC
        if "prompt" in str(filepath):
            artifact_type = ArtifactType.PROMPT
        elif "newsletter" in str(filepath) or "Fortress" in str(filepath):
            artifact_type = ArtifactType.NEWSLETTER

        metadata = GoldStandardScrubber.extract_metadata(text, artifact_type.value)

        # Vector Database / RAG embedding prep
        metadata["semantic_chunks"] = len(chunks)
        metadata["chunk_preview"] = chunks[0][:100] + "..." if chunks else ""

        artifact = GoldStandardArtifact(
            source_path=str(filepath),
            content={"full_text": text, "chunks": chunks},
            artifact_type=artifact_type,
            title=title,
            metadata=metadata
        )
        self.artifacts.append(artifact)

    async def _process_text_async(self, filepath: Path):
        async with aiofiles.open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            raw_text = await f.read()
            text = GoldStandardScrubber.clean_text(raw_text)

        # Semantic Chunking via ChunkingEngine
        chunk_dicts = self.chunking_engine.chunk(text)
        chunks = [c["text"] for c in chunk_dicts if "text" in c]

        metadata = GoldStandardScrubber.extract_metadata(text, ArtifactType.UNKNOWN.value)
        metadata["semantic_chunks"] = len(chunks)
        metadata["chunk_preview"] = chunks[0][:100] + "..." if chunks else ""

        artifact = GoldStandardArtifact(
            source_path=str(filepath),
            content={"full_text": text, "chunks": chunks},
            artifact_type=ArtifactType.UNKNOWN,
            title=filepath.name,
            metadata=metadata
        )
        self.artifacts.append(artifact)

    async def _process_python_async(self, filepath: Path):
        try:
            async with aiofiles.open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                source = await f.read()

            tree = ast.parse(source)
            docstring = ast.get_docstring(tree)

            classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
            functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]

            content = f"Module: {filepath.name}\n"
            if docstring:
                content += f"Docstring: {docstring}\n"
            if classes:
                content += f"Classes: {', '.join(classes)}\n"
            if functions:
                content += f"Functions: {', '.join(functions)}\n"

            metadata = {
                "classes": classes,
                "functions": functions,
                "has_docstring": bool(docstring)
            }

            artifact = GoldStandardArtifact(
                source_path=str(filepath),
                content=content,
                artifact_type=ArtifactType.CODE_DOC,
                title=f"Doc: {filepath.name}",
                metadata=metadata
            )
            self.artifacts.append(artifact)

        except Exception:
            pass

    def save_to_jsonl(self, output_path: str):
        """Saves all artifacts to a JSONL file safely handling directories."""
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open('w', encoding='utf-8') as f:
            for artifact in self.artifacts:
                f.write(json.dumps(artifact.to_dict()) + '\n')
        print(f"Saved {len(self.artifacts)} artifacts to {output_path}")

    def get_artifacts_by_type(self, artifact_type: ArtifactType) -> list[dict]:
        return [a.to_dict() for a in self.artifacts if a.type == artifact_type.value]


if __name__ == "__main__":
    ingestor = UniversalIngestor()
    ingestor.scan_directory("core")  # Scan core for code docs
    ingestor.scan_directory("prompt_library")
    ingestor.scan_directory("data")
    ingestor.scan_directory("docs")

    ingestor.save_to_jsonl("data/gold_standard/knowledge_artifacts.jsonl")
