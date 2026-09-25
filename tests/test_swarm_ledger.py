from scripts.swarm_ledger import SwarmMessage, AppendOnlyLedger, KVStateStore

def test_ledger_append():
    ledger = AppendOnlyLedger()
    msg = SwarmMessage(message_id="msg1", sender="agentA", payload={"data": 1}, timestamp=123)
    ledger.append(msg)
    assert len(ledger.entries) == 1
    assert ledger.entries[0].message_id == "msg1"

def test_hash_cache():
    ledger = AppendOnlyLedger()
    msg = SwarmMessage(message_id="msg1", sender="agentA", payload={"data": 1}, timestamp=123)
    ledger.append(msg)
    h = ledger.hash_cache()
    assert isinstance(h, str)
    assert len(h) == 64

def test_kv_store():
    store = KVStateStore()
    store.set_state("key1", {"val": "hello"})
    assert store.get_state("key1") == {"val": "hello"}
    assert store.get_state("key2") is None
