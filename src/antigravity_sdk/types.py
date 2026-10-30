"""
Core data types and multimodal content abstractions for Antigravity SDK.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Content:
    """Base class for multimodal prompt content attachments."""
    description: str | None = None


@dataclass
class Image(Content):
    """Image attachment wrapper for in-memory bytes or file references."""
    data: bytes | None = None
    mime_type: str = "image/png"
    uri: str | None = None


@dataclass
class Document(Content):
    """Document attachment wrapper (PDF, text, markdown, etc.)."""
    data: bytes | None = None
    mime_type: str = "application/pdf"
    uri: str | None = None
    file_path: str | None = None


def from_file(
    file_path: str | Path,
    mime_type: str | None = None,
    description: str | None = None,
) -> Content:
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
        return Document(
            data=content_bytes,
            mime_type=mime_type,
            uri=str(path.resolve()),
            file_path=str(path),
            description=description,
        )


@dataclass
class McpStdioServer:
    """Configuration for an MCP (Model Context Protocol) STDIO server connection."""
    name: str
    command: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)


@dataclass
class ToolCall:
    """Strongly typed tool call dispatch event."""
    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass
class Step:
    """Represents a discrete turn/step execution in a Conversation."""
    step_id: str
    content: str
    role: str = "assistant"
    is_complete_response: bool = True
    tool_calls: list[ToolCall] = field(default_factory=list)
    thoughts: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
