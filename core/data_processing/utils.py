import os
import json
import uuid
import hashlib
import ast
import re
import logging
from typing import List, Dict, Any, Optional, Union
from enum import Enum
from datetime import datetime
from dataclasses import dataclass, field, asdict
import pandas as pd

logger = logging.getLogger("DataUtils")

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

    @staticmethod
    def handle_parquet(filepath: str, raw_text: str = None) -> Optional[GoldStandardArtifact]:
        try:
            df = pd.read_parquet(filepath)

            # Create a summary
            summary = f"Parquet Data: {os.path.basename(filepath)}\n"
            summary += f"Columns: {', '.join(df.columns)}\n"
            summary += f"Rows: {len(df)}\n"

            if not df.empty:
                summary += f"Date Range: {df.index.min()} to {df.index.max()}\n"
                meta = {
                    "columns": list(df.columns),
                    "row_count": len(df),
                    "start_date": str(df.index.min()),
                    "end_date": str(df.index.max())
                }
            else:
                summary += "Empty DataFrame\n"
                meta = {
                    "columns": list(df.columns),
                    "row_count": 0
                }

            return GoldStandardArtifact(
                source_path=filepath,
                content=summary,
                artifact_type=ArtifactType.DATA.value,
                title=f"Market Data: {os.path.basename(filepath)}",
                metadata=meta,
                content_hash=hashlib.md5(filepath.encode()).hexdigest()
            )
        except Exception as e:
            logger.error(f"Parquet Read Error {filepath}: {e}")
            return None
