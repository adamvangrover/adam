"""
Context Cache Lifecycle Manager & Unit Economics Engine (F-04, Gate G3).

Implements bounded context cache lifecycle management:
- Deal room tenant isolation
- Context version binding
- Explicit TTL (Time-To-Live) expiration
- LRU (Least-Recently-Used) capacity eviction
- Reconciliation of estimated pricing vs. observed provider telemetry
"""

import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any


@dataclass
class CachePricingModel:
    initial_write_cost_per_1m: float = 2.00
    cached_read_cost_per_1m: float = 0.10
    output_cost_per_1m: float = 10.00
    uncached_read_cost_per_1m: float = 2.00
    competitor_read_cost_per_1m: float = 5.00
    competitor_output_cost_per_1m: float = 15.00


@dataclass
class DealRoomCacheEntry:
    deal_room_id: str
    context_version: str
    token_count: int
    data: Any
    created_at: float = field(default_factory=time.time)
    last_accessed_at: float = field(default_factory=time.time)
    hit_count: int = 0


class DealRoomContextCache:
    """
    Bounded, tenant-isolated context cache manager with TTL, LRU eviction, and telemetry tracking.
    """

    def __init__(
        self,
        max_capacity_entries: int = 100,
        ttl_seconds: float = 3600.0,
        pricing_model: CachePricingModel | None = None,
    ) -> None:
        self.max_capacity = max_capacity_entries
        self.ttl_seconds = ttl_seconds
        self.pricing = pricing_model or CachePricingModel()
        # Storage: deal_room_id -> OrderedDict[context_version, DealRoomCacheEntry]
        self._store: dict[str, OrderedDict[str, DealRoomCacheEntry]] = {}
        # Observed telemetry metrics
        self.total_cache_hits = 0
        self.total_cache_misses = 0
        self.total_evictions = 0
        self.total_expirations = 0

    def get(self, deal_room_id: str, context_version: str) -> DealRoomCacheEntry | None:
        """
        Retrieves an active, unexpired entry from the cache for a specific deal room and version.
        """
        if deal_room_id not in self._store or context_version not in self._store[deal_room_id]:
            self.total_cache_misses += 1
            return None

        entry = self._store[deal_room_id][context_version]
        now = time.time()

        # Check TTL expiration
        if now - entry.created_at > self.ttl_seconds:
            self.total_expirations += 1
            self.total_cache_misses += 1
            del self._store[deal_room_id][context_version]
            if not self._store[deal_room_id]:
                del self._store[deal_room_id]
            return None

        # Update LRU ordering and access metadata
        self._store[deal_room_id].move_to_end(context_version)
        entry.last_accessed_at = now
        entry.hit_count += 1
        self.total_cache_hits += 1
        return entry

    def put(
        self,
        deal_room_id: str,
        context_version: str,
        token_count: int,
        data: object,
    ) -> DealRoomCacheEntry:
        """
        Stores an entry in the cache with tenant isolation and LRU capacity enforcement.
        """
        now = time.time()
        entry = DealRoomCacheEntry(
            deal_room_id=deal_room_id,
            context_version=context_version,
            token_count=token_count,
            data=data,
            created_at=now,
            last_accessed_at=now,
        )

        if deal_room_id not in self._store:
            self._store[deal_room_id] = OrderedDict()

        # Check capacity across all entries
        total_entries = sum(len(versions) for versions in self._store.values())
        if total_entries >= self.max_capacity and context_version not in self._store[deal_room_id]:
            self._evict_lru()

        self._store[deal_room_id][context_version] = entry
        self._store[deal_room_id].move_to_end(context_version)
        return entry

    def _evict_lru(self) -> None:
        """Evicts the least recently accessed entry across all deal rooms."""
        oldest_entry: tuple[str, str, float] | None = None

        for room_id, versions in self._store.items():
            for v_id, entry in versions.items():
                if oldest_entry is None or entry.last_accessed_at < oldest_entry[2]:
                    oldest_entry = (room_id, v_id, entry.last_accessed_at)

        if oldest_entry:
            room_id, v_id, _ = oldest_entry
            del self._store[room_id][v_id]
            if not self._store[room_id]:
                del self._store[room_id]
            self.total_evictions += 1

    def calculate_deal_room_economics(
        self,
        input_tokens_per_pass: int = 850_000,
        passes: int = 12,
        output_tokens: int = 48_000,
    ) -> dict[str, float]:
        """
        Reproduces and calculates deal room economics for context caching scenario.
        """
        # Initial context write (1 pass at full rate)
        initial_write_cost = (input_tokens_per_pass / 1_000_000) * self.pricing.initial_write_cost_per_1m
        # 11 cached reads at discounted rate
        cached_read_tokens = input_tokens_per_pass * (passes - 1)
        cached_reads_cost = (cached_read_tokens / 1_000_000) * self.pricing.cached_read_cost_per_1m
        # Output generation
        output_cost = (output_tokens / 1_000_000) * self.pricing.output_cost_per_1m

        cached_total = initial_write_cost + cached_reads_cost + output_cost

        # Uncached baseline (12 full-rate input passes + output)
        total_input_tokens = input_tokens_per_pass * passes
        uncached_baseline = (total_input_tokens / 1_000_000) * self.pricing.uncached_read_cost_per_1m + output_cost

        # Competitor baseline
        competitor_baseline = (
            (total_input_tokens / 1_000_000) * self.pricing.competitor_read_cost_per_1m
            + (output_tokens / 1_000_000) * self.pricing.competitor_output_cost_per_1m
        )

        savings_vs_uncached_pct = ((uncached_baseline - cached_total) / uncached_baseline) * 100
        savings_vs_competitor_pct = ((competitor_baseline - cached_total) / competitor_baseline) * 100

        return {
            "initial_context_write_cost": round(initial_write_cost, 4),
            "cached_reads_cost": round(cached_reads_cost, 4),
            "output_generation_cost": round(output_cost, 4),
            "cached_total_cost": round(cached_total, 4),
            "uncached_baseline_cost": round(uncached_baseline, 4),
            "competitor_baseline_cost": round(competitor_baseline, 4),
            "savings_vs_uncached_pct": round(savings_vs_uncached_pct, 2),
            "savings_vs_competitor_pct": round(savings_vs_competitor_pct, 2),
        }
