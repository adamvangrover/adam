import time
from core.remediation.cache_manager import DealRoomContextCache

def test_cache_put_and_get() -> None:
    cache = DealRoomContextCache(max_capacity_entries=5, ttl_seconds=3600.0)
    entry = cache.put("DEAL_001", "v1.0", 850_000, {"data": "room_content"})
    assert entry.deal_room_id == "DEAL_001"

    retrieved = cache.get("DEAL_001", "v1.0")
    assert retrieved is not None
    assert retrieved.hit_count == 1
    assert cache.total_cache_hits == 1

def test_cache_tenant_isolation() -> None:
    cache = DealRoomContextCache()
    cache.put("DEAL_001", "v1.0", 850_000, {"data": "room1"})
    cache.put("DEAL_002", "v1.0", 850_000, {"data": "room2"})

    entry1 = cache.get("DEAL_001", "v1.0")
    entry2 = cache.get("DEAL_002", "v1.0")
    assert entry1 is not None and entry1.data["data"] == "room1"
    assert entry2 is not None and entry2.data["data"] == "room2"
    assert cache.get("DEAL_001", "v2.0") is None

def test_cache_ttl_expiration() -> None:
    cache = DealRoomContextCache(ttl_seconds=0.1)
    cache.put("DEAL_001", "v1.0", 850_000, {"data": "room_content"})
    time.sleep(0.15)
    assert cache.get("DEAL_001", "v1.0") is None
    assert cache.total_expirations == 1

def test_cache_lru_eviction() -> None:
    cache = DealRoomContextCache(max_capacity_entries=2)
    cache.put("DEAL_001", "v1.0", 850_000, "data1")
    time.sleep(0.01)
    cache.put("DEAL_002", "v1.0", 850_000, "data2")
    time.sleep(0.01)
    # Put 3rd item, should evict DEAL_001
    cache.put("DEAL_003", "v1.0", 850_000, "data3")

    assert cache.get("DEAL_001", "v1.0") is None
    assert cache.get("DEAL_002", "v1.0") is not None
    assert cache.get("DEAL_003", "v1.0") is not None
    assert cache.total_evictions == 1

def test_deal_room_economics_reproduction() -> None:
    cache = DealRoomContextCache()
    econ = cache.calculate_deal_room_economics(
        input_tokens_per_pass=850_000,
        passes=12,
        output_tokens=48_000,
    )
    assert econ["initial_context_write_cost"] == 1.700
    assert econ["cached_reads_cost"] == 0.935
    assert econ["output_generation_cost"] == 0.480
    assert econ["cached_total_cost"] == 3.115
    assert econ["uncached_baseline_cost"] == 20.880
    assert econ["competitor_baseline_cost"] == 51.720
    assert econ["savings_vs_uncached_pct"] == 85.08
    assert econ["savings_vs_competitor_pct"] == 93.98
