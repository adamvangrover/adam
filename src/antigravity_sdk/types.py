"""
Core data types and multimodal content abstractions for Antigravity SDK.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


@dataclass
class Content:
    """Base class for multimodal prompt content attachments."""
    description: Optional[str] = None


@dataclass
class Image(Content):
    """Image attachment wrapper for in-memory bytes or file references."""
    data: Optional[bytes] = None
    mime_type: str = "image/png"
    uri: Optional[str] = None


@dataclass
class Document(Content):
    """Document attachment wrapper (PDF, text, markdown, etc.)."""
    data: Optional[bytes] = None
    mime_type: str = "application/pdf"
    uri: Optional[str] = None
    file_path: Optional[str] = None


def from_file(file_path: Union[str, Path], mime_type: Optional[str] = None, description: Optional[str] = None) -> Content:
    """
    Convenience helper to auto-detect and resolve a file from disk into a Document or Image.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found at path: {file_path}")

    ext = path.suffix.lower()
    if mime_type is None:
        if ext in [".png", ".jpg", ".jpeg", ".webp", ".gif"]:
            mime_type = f"image/{ext[1:]}"
            if ext == ".jpg":
                mime_type = "image/jpeg"
        elif ext == ".pdf":
            mime_type = "application/pdf"
        elif ext in [".txt", ".md", ".py", ".json"]:
            mime_type = "text/plain"
        else:
            mime_type = "application/octet-stream"

    content_bytes = path.read_bytes()
    if mime_type.startswith("image/"):
        return Image(data=content_bytes, mime_type=mime_type, uri=str(path.resolve()), description=description)
    else:
        return Document(data=content_bytes, mime_type=mime_type, uri=str(path.resolve()), file_path=str(path), description=description)


@dataclass
class McpStdioServer:
    """Configuration for an MCP (Model Context Protocol) STDIO server connection."""
    name: str
    command: str
    args: List[str] = field(default_factory=list)
    env: Dict[str, str] = field(default_factory=dict)


@dataclass
class ToolCall:
    """Strongly typed tool call dispatch event."""
    id: str
    name: str
    arguments: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Step:
    """Represents a discrete turn/step execution in a Conversation."""
    step_id: str
    content: str
    role: str = "assistant"
    is_complete_response: bool = True
    tool_calls: List[ToolCall] = field(default_factory=list)
    thoughts: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
