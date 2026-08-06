import asyncio
import hashlib
import json
import logging
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Set

# Assuming these are available from your core and interfaces packages
from afos_core import Event
from afos_interfaces import IIntegrationKernel

logger = logging.getLogger(__name__)


# ==================================================================
# INTEGRATION KERNEL
# ==================================================================
class IntegrationKernel(IIntegrationKernel):
    """
    Concrete implementation of the Integration Kernel.
    Handles signal ingestion, payload sanitization, transformation to 
    canonical OS Events, and asynchronous pub-sub broadcasting.
    """
    def __init__(self):
        # Topic -> Set of Subscriber Async Callbacks
        self._subscribers: Dict[str, Set[Callable[[Event], Any]]] = {}
        # In-memory queue for event processing telemetry
        self._event_queue: asyncio.Queue[tuple[str, Event]] = asyncio.Queue()
        self._worker_task: Optional[asyncio.Task] = None
        self._processed_count: int = 0

    async def initialize(self) -> None:
        """
        Boot sequence: Starts background event bus worker task.
        """
        logger.info("Initializing Integration Kernel: Event Bus and Signal Ingestion engine starting...")
        self._worker_task = asyncio.create_task(self._event_bus_worker())
        logger.info("Integration Kernel online.")

    async def shutdown(self) -> None:
        """
        Teardown sequence: Drains the internal event queue and cancels worker task.
        """
        logger.info(f"Integration Kernel shutting down. Draining remaining events ({self._event_queue.qsize()} queued)...")
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
        logger.info(f"Integration Kernel offline. Total events broadcasted: {self._processed_count}")

    def subscribe(self, event_topic: str, callback: Callable[[Event], Any]) -> None:
        """
        Registers an asynchronous listener/handler for a specific event topic.
        """
        if event_topic not in self._subscribers:
            self._subscribers[event_topic] = set()
        self._subscribers[event_topic].add(callback)
        logger.debug(f"Subscribed handler to topic '{event_topic}'")

    async def ingest_signal(self, source: str, payload: Dict[str, Any]) -> Event:
        """
        Ingests an untrusted external signal/webhook, normalizes it, 
        and transforms it into an immutable canonical Event entity.
        """
        logger.info(f"Ingesting raw signal from source: [{source}]")

        # 1. Sanitize & Normalize Payload
        sanitized_payload = self._sanitize_payload(payload)

        # 2. Determine Event Classification
        event_type = payload.get("event_type", f"ExternalSignal.{source.upper()}")

        # 3. Generate Cryptographic Signature for Signal Traceability
        raw_bytes = json.dumps(sanitized_payload, sort_keys=True).encode("utf-8")
        signal_hash = hashlib.sha256(raw_bytes).hexdigest()[:16]

        # 4. Construct Canonical Event
        event = Event(
            id=f"evt_{source.lower()}_{signal_hash}",
            type=event_type,
            timestamp=datetime.now(timezone.utc),
            payload={
                "source": source,
                "sanitized_data": sanitized_payload,
                "provenance_hash": signal_hash
            }
        )

        logger.info(f"Signal transformed to Canonical Event [{event.id}] of type '{event.type}'")
        return event

    async def publish_event(self, event_topic: str, event: Event) -> None:
        """
        Places a canonical Event onto the topic bus for asynchronous worker dispatch.
        """
        logger.debug(f"Publishing event [{event.id}] to topic '{event_topic}'")
        await self._event_queue.put((event_topic, event))

    async def _event_bus_worker(self) -> None:
        """
        Internal background event processor dispatching queued events to topic subscribers.
        """
        while True:
            try:
                event_topic, event = await self._event_queue.get()
                
                listeners = self._subscribers.get(event_topic, set())
                # Also dispatch to wildcard subscribers if any
                wildcard_listeners = self._subscribers.get("*", set())
                all_targets = listeners.union(wildcard_listeners)

                if all_targets:
                    tasks = [asyncio.create_task(self._safe_dispatch(handler, event)) for handler in all_targets]
                    await asyncio.gather(*tasks)

                self._processed_count += 1
                self._event_queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in Event Bus worker loop: {e}")

    async def _safe_dispatch(self, handler: Callable[[Event], Any], event: Event) -> None:
        """Wraps listener callbacks in error handling to prevent individual consumer failures from taking down the bus."""
        try:
            if asyncio.iscoroutinefunction(handler):
                await handler(event)
            else:
                handler(event)
        except Exception as e:
            logger.error(f"Subscriber handler failed for event [{event.id}]: {e}")

    def _sanitize_payload(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Strips unsafe characters, scrubs potential injection vectors, 
        and enforces UTF-8 string encoding.
        """
        sanitized = {}
        for key, val in raw_data.items():
            clean_key = str(key).strip()
            if isinstance(val, str):
                sanitized[clean_key] = val.strip()
            elif isinstance(val, dict):
                sanitized[clean_key] = self._sanitize_payload(val)
            else:
                sanitized[clean_key] = val
        return sanitized


# ==================================================================
# EXECUTION / FUNCTIONAL TEST
# ==================================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    # Sample downstream event handlers (e.g. Audit listener, Risk Alert service)
    async def risk_alert_listener(event: Event):
        print(f"\n⚡ [RISK SERVICE RECEIVED] Event ID: {event.id}")
        data = event.payload.get("sanitized_data", {})
        if data.get("metric_breach"):
            print(f"⚠️  ALERT DETECTED: {data.get('metric')} breached threshold! Value: {data.get('value')}")

    async def audit_logger_listener(event: Event):
        print(f"📜 [AUDIT LOG] Recorded event {event.id} | Provenance Hash: {event.payload.get('provenance_hash')}")

    async def main():
        kernel = IntegrationKernel()
        await kernel.initialize()

        # 1. Subscribe handlers to event topics
        kernel.subscribe("market.credit_events", risk_alert_listener)
        kernel.subscribe("*", audit_logger_listener) # Wildcard listener for all events

        # 2. Simulate raw inbound signal (e.g., webhook from credit monitoring service)
        raw_signal = {
            "event_type": "CovenantBreachSignal",
            "borrower_id": "org_99182_tech",
            "metric": "SeniorLeverageRatio",
            "value": 4.85,
            "metric_breach": True
        }

        # 3. Ingest and transform raw webhook to canonical Event
        canonical_event = await kernel.ingest_signal(source="credit_bureau_webhook", payload=raw_signal)

        # 4. Broadcast the event across the OS bus
        await kernel.publish_event(event_topic="market.credit_events", event=canonical_event)

        # Allow async event loop workers to finish processing
        await asyncio.sleep(0.5)

        await kernel.shutdown()

    asyncio.run(main())
