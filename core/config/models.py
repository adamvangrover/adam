"""
Model Specifications and Router Configuration under SR 11-7 Model Risk Governance.
"""

from typing import Any, Optional
import yaml
from pydantic import BaseModel, field_validator


class SR11_7_BLOCKED_MODEL_ERROR(Exception):
    """Raised when a quarantined or unaligned model is instantiated."""
    pass


BLOCKED_MODELS = {"gpt-6.1-astra", "gpt-6.1-astra-unaligned"}


class PricingSpec(BaseModel):
    currency: str = "USD"
    base_input_per_million: float = 2.00
    cached_read_per_million: float = 0.10
    output_per_million: float = 10.00
    cache_discount_pct: float = 95.0


class ModelSpec(BaseModel):
    provider: str
    model_id: str
    role: Optional[str] = None
    release_date: Optional[str] = None
    architecture: Optional[str] = None
    context_window: Optional[int] = None
    pricing: Optional[PricingSpec] = None
    capabilities: Optional[list[str]] = None
    governance: Optional[dict[str, Any]] = None

    @field_validator("model_id", mode="before")
    @classmethod
    def validate_model_not_quarantined(cls, v: str) -> str:
        if isinstance(v, str):
            clean_v = v.strip().lower()
            if clean_v in BLOCKED_MODELS or "astra" in clean_v:
                raise SR11_7_BLOCKED_MODEL_ERROR(
                    f"SR 11-7 VIOLATION: Model '{v}' is permanently quarantined. "
                    "Permission-bypassing and unaligned autonomous behavior detected in October 2026 cohort."
                )
        return v


class DealRoomCacheConfig(BaseModel):
    default_ttl_seconds: int = 7200
    eviction_policy: str = "LRU"
    max_cached_deal_rooms: int = 16
    min_tokens_for_cache: int = 32768


class Tier1PrimarySystem2Spec(BaseModel):
    provider: str = "google"
    model_id: str = "gemini-4-argon"
    release_date: Optional[str] = None
    architecture: Optional[str] = None
    context_window: Optional[int] = None
    pricing: Optional[PricingSpec] = None
    capabilities: Optional[list[str]] = None
    governance: Optional[dict[str, Any]] = None


class RouterConfig(BaseModel):
    version: Optional[str] = "2026.10"
    schema_version: Optional[str] = "2.0.0"
    deal_room_cache: Optional[DealRoomCacheConfig] = None
    tier_1_primary_system2: Optional[Tier1PrimarySystem2Spec] = None
    routing_tiers: Optional[dict[str, Any]] = None

    @classmethod
    def load_from_yaml(cls, path: str) -> "RouterConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        routing_tiers = data.get("routing_tiers", {})
        t1_data = routing_tiers.get("tier_1_primary_system2", {})
        if t1_data and "pricing" in t1_data and isinstance(t1_data["pricing"], dict):
            t1_data = dict(t1_data)
            t1_data["pricing"] = PricingSpec(**t1_data["pricing"])
        tier_1 = Tier1PrimarySystem2Spec(**t1_data) if t1_data else None

        cache_data = data.get("deal_room_cache", {})
        deal_room_cache = DealRoomCacheConfig(**cache_data) if cache_data else None

        return cls(
            version=data.get("version"),
            schema_version=data.get("schema_version"),
            deal_room_cache=deal_room_cache,
            tier_1_primary_system2=tier_1,
            routing_tiers=routing_tiers,
        )
