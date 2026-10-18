"""
PROV-O Telemetry and Negative Finding Reporting Module.
"""

from typing import Any, ClassVar
from pydantic import BaseModel, Field


class UnverifiedNegativeFinding(BaseModel):
    TOKEN: ClassVar[str] = "UNVERIFIED_NEGATIVE_FINDING"

    target_metric: str
    source_document: str = "Form 10-K FY2026"
    accession_number: str = "0000320193-26-000102"
    searched_sections: list[str] = Field(
        default_factory=lambda: ["Item 7 - MD&A", "Note 14 - Debt and Financing"]
    )
    exhaustion_provenance: str = (
        "Full token horizon search exhausted. No statutory schedule or audited footnote found."
    )
    token: str = "UNVERIFIED_NEGATIVE_FINDING"

    def to_dict(self) -> dict[str, Any]:
        return {
            "token": self.token,
            "target_metric": self.target_metric,
            "source_document": self.source_document,
            "accession_number": self.accession_number,
            "searched_sections": self.searched_sections,
            "exhaustion_provenance": self.exhaustion_provenance,
        }


class LogfireSpanContextManager:
    def __init__(self, name: str, **kwargs: Any) -> None:
        self.name = name
        self.kwargs = kwargs

    def __enter__(self) -> "LogfireSpanContextManager":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        pass


class LogfireTelemetry:
    def span(self, name: str, **kwargs: Any) -> LogfireSpanContextManager:
        return LogfireSpanContextManager(name, **kwargs)


logfire = LogfireTelemetry()
