import hashlib
import time
from typing import Dict, List, Optional, Any


class DealRoomContextManager:
    """
    Manages deal-room context with canonical document snapshot store,
    content-addressed 2000-character chunking with SHA-256 locators,
    provider-side prompt caching simulation with TTL eviction, and token billing reconciliation.
    """

    def __init__(self, cache_ttl_seconds: float = 300.0, chunk_size: int = 2000):
        self.cache_ttl_seconds = cache_ttl_seconds
        self.chunk_size = chunk_size
        self.snapshots: Dict[str, str] = {}  # doc_id -> raw content
        self.chunks: Dict[str, str] = {}     # chunk_hash -> chunk_text
        self.doc_chunks: Dict[str, List[str]] = {}  # doc_id -> list of chunk_hashes
        self.cached_context: Dict[str, float] = {}  # context_hash -> timestamp created/refreshed
        self.billing_log: List[Dict[str, Any]] = []

    def ingest_source_document(self, doc_id: str, content: str) -> str:
        doc_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        self.snapshots[doc_id] = content

        # Content-addressed chunking
        chunk_hashes = []
        for i in range(0, len(content), self.chunk_size):
            chunk_text = content[i : i + self.chunk_size]
            c_hash = hashlib.sha256(chunk_text.encode("utf-8")).hexdigest()
            self.chunks[c_hash] = chunk_text
            chunk_hashes.append(c_hash)

        self.doc_chunks[doc_id] = chunk_hashes
        return doc_hash

    def evaluate_pass_with_cache(
        self,
        context_key: str,
        input_tokens: int,
        output_tokens: int,
        base_rate_per_m: float = 2.00,
        cached_rate_per_m: float = 0.10,
        output_rate_per_m: float = 10.00,
        now: Optional[float] = None
    ) -> Dict[str, Any]:
        current_time = now if now is not None else time.time()

        # Check cache validity
        is_cache_hit = False
        if context_key in self.cached_context:
            created_at = self.cached_context[context_key]
            if current_time - created_at <= self.cache_ttl_seconds:
                is_cache_hit = True
            else:
                # Expired
                del self.cached_context[context_key]

        if is_cache_hit:
            uncached_tokens = 0
            cached_tokens = input_tokens
            # Refresh TTL on cache hit
            self.cached_context[context_key] = current_time
        else:
            uncached_tokens = input_tokens
            cached_tokens = 0
            # Write to cache
            self.cached_context[context_key] = current_time

        cost_uncached_input = (uncached_tokens / 1_000_000.0) * base_rate_per_m
        cost_cached_input = (cached_tokens / 1_000_000.0) * cached_rate_per_m
        cost_output = (output_tokens / 1_000_000.0) * output_rate_per_m
        total_cost = cost_uncached_input + cost_cached_input + cost_output

        billing_record = {
            "context_key": context_key,
            "timestamp": current_time,
            "is_cache_hit": is_cache_hit,
            "uncached_input_tokens": uncached_tokens,
            "cached_input_tokens": cached_tokens,
            "output_tokens": output_tokens,
            "total_cost_usd": round(total_cost, 6)
        }
        self.billing_log.append(billing_record)
        return billing_record

    def get_aggregate_billing_summary(self) -> Dict[str, Any]:
        total_cost = sum(r["total_cost_usd"] for r in self.billing_log)
        total_uncached = sum(r["uncached_input_tokens"] for r in self.billing_log)
        total_cached = sum(r["cached_input_tokens"] for r in self.billing_log)
        total_output = sum(r["output_tokens"] for r in self.billing_log)
        cache_hits = sum(1 for r in self.billing_log if r["is_cache_hit"])
        total_passes = len(self.billing_log)

        return {
            "total_passes": total_passes,
            "cache_hits": cache_hits,
            "total_uncached_input_tokens": total_uncached,
            "total_cached_input_tokens": total_cached,
            "total_output_tokens": total_output,
            "total_cost_usd": round(total_cost, 6)
        }
