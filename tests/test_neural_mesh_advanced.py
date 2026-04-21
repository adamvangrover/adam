import pytest
import asyncio
from core.v30_architecture.python_intelligence.bridge.ephemeral_cortex import EphemeralCortex
from core.v30_architecture.python_intelligence.bridge.neural_mesh import TopicRouter, NeuralPacket

@pytest.mark.asyncio
async def test_ephemeral_cortex_ingest_and_query():
    cortex = EphemeralCortex(max_history_per_topic=2)
    packet1 = NeuralPacket(source_agent="Test", packet_type="test_topic", payload={"data": 1})
    packet2 = NeuralPacket(source_agent="Test", packet_type="test_topic", payload={"data": 2})
    packet3 = NeuralPacket(source_agent="Test", packet_type="test_topic", payload={"data": 3})

    await cortex.ingest("test_topic", packet1)
    await cortex.ingest("test_topic", packet2)
    await cortex.ingest("test_topic", packet3)

    # Should only keep the last 2 due to max_history_per_topic=2
    history = await cortex.query("test_topic")
    assert len(history) == 2
    assert history[0].payload["data"] == 2
    assert history[1].payload["data"] == 3

@pytest.mark.asyncio
async def test_ephemeral_cortex_flush():
    cortex = EphemeralCortex()
    packet = NeuralPacket(source_agent="Test", packet_type="test_topic", payload={"data": 1})
    await cortex.ingest("test_topic", packet)

    await cortex.flush()
    history = await cortex.query("test_topic")
    assert len(history) == 0

class DummyWebSocket:
    def __init__(self, id):
        self.id = id

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return self.id == other.id

@pytest.mark.asyncio
async def test_topic_router():
    router = TopicRouter()
    ws1 = DummyWebSocket(1)
    ws2 = DummyWebSocket(2)

    await router.subscribe(ws1, "topic_A")
    await router.subscribe(ws2, "topic_A")
    await router.subscribe(ws2, "topic_B")

    subs_A = await router.get_subscribers("topic_A")
    assert len(subs_A) == 2
    assert ws1 in subs_A
    assert ws2 in subs_A

    subs_B = await router.get_subscribers("topic_B")
    assert len(subs_B) == 1
    assert ws2 in subs_B

    await router.unsubscribe(ws1, "topic_A")
    subs_A_after = await router.get_subscribers("topic_A")
    assert len(subs_A_after) == 1
    assert ws2 in subs_A_after
