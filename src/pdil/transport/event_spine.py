import uuid
import hashlib
import json
from typing import Dict, Any, Optional
from datetime import datetime, timezone

class EventEnvelope:
    """
    Defines the standard event envelope for state-changing operations transported over the event spine.
    Ensures idempotency and includes the frozen context hash.
    """
    def __init__(self, payload: Dict[str, Any], context_hash: str, stream_subject: str):
        self.payload = payload
        self.context_hash = context_hash
        self.stream_subject = stream_subject
        self.timestamp = datetime.now(timezone.utc).isoformat()

        # Generates a deterministic message ID based on payload and context to allow deduplication windows
        canonical_json = json.dumps({"payload": payload, "context": context_hash}, sort_keys=True).encode('utf-8')
        self.msg_id = hashlib.sha256(canonical_json).hexdigest()

    def serialize(self) -> Dict[str, Any]:
        return {
            "msg_id": self.msg_id,
            "subject": self.stream_subject,
            "timestamp": self.timestamp,
            "context_hash": self.context_hash,
            "payload": self.payload
        }

class EventSpineSimulation:
    """
    Simulates the NATS JetStream event spine deduplication window functionality.
    """
    def __init__(self):
        self.seen_messages = set()

    def publish(self, envelope: EventEnvelope) -> bool:
        """
        Publishes an event to the spine. Returns False if silently discarded due to deduplication.
        """
        if envelope.msg_id in self.seen_messages:
            # Silently discard duplicates (NATS JetStream exactly-once semantics)
            return False

        self.seen_messages.add(envelope.msg_id)
        # In a real implementation, this would publish to NATS
        return True
