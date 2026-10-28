"""
AFOS V3 Triangulated Consensus Engine & Fortress/Hunt State Manager
Phase 2 (3-Node Concurrent Fetch & 0.5-Sigma Outlier Rejection) &
Phase 4 (LOCKDOWN Circuit Breaker preserving 75% Fortress / 25% Convex Hunt Kernel)
"""

from __future__ import annotations

import asyncio
import hmac
import statistics
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Callable, Coroutine, Final

from afos_v3.gateway.zero_trust_gateway import (
    EphemeralSTSToken,
    MTLSConnectionHandshake,
    QuarantineSink,
    ZeroTrustGatewayMiddleware,
    ZeroTrustSecurityError,
)
from afos_v3.validation.schema_bounds import CoreMetricsPayload, DeterministicValidationEngine

FORTRESS_WEIGHT: Final[float] = 0.75  # 75% Fortress (sub-90-day sovereign paper)
CONVEX_HUNT_WEIGHT: Final[float] = 0.25  # 25% Convex Hunt
OUTLIER_SIGMA_THRESHOLD: Final[float] = 0.5  # Strict 0.5 standard deviation threshold
MIN_VERIFIED_QUORUM: Final[int] = 2  # Minimum 2 out of 3 verified feeds required
MICROSTRUCTURE_REL_NOISE_FLOOR: Final[float] = 1e-3  # 10 bps tick noise floor for identical feeds

METRIC_FIELDS: Final[tuple[str, ...]] = (
    "sp500_price",
    "us10y_yield",
    "vix",
    "btc_usd",
    "brent_crude",
)


class SystemStatus(StrEnum):
    NOMINAL = "NOMINAL"
    LOCKDOWN = "LOCKDOWN"


class ConsensusLockdownError(ZeroTrustSecurityError):
    """Raised when consensus falls below quorum and SYSTEM_STATUS enters LOCKDOWN."""


@dataclass(frozen=True, slots=True)
class CapitalAllocationDirective:
    """Output of the AFOS V3 Kernel 75/25 Portfolio State Router."""

    system_status: SystemStatus
    total_nav: float
    fortress_sovereign_paper_nav: float
    fortress_weight: float
    convex_hunt_active_nav: float
    convex_hunt_standby_cash_nav: float
    convex_hunt_execution_enabled: bool
    lockdown_reason: str | None = None


class AFOSKernelStateManager:
    """
    Phase 4.2: Manages AFOS V3 Kernel state while preserving the core
    75% Fortress / 25% Convex Hunt routing logic.
    Automatically halts the 25% Convex Hunt execution leg under LOCKDOWN
    until manual meatspace cryptographic authorization is supplied.
    """

    def __init__(self, meatspace_auth_secret: bytes) -> None:
        self._status: SystemStatus = SystemStatus.NOMINAL
        self._lockdown_reason: str | None = None
        self._severed_nodes: set[str] = set()
        self._meatspace_secret: bytes = meatspace_auth_secret

    @property
    def SYSTEM_STATUS(self) -> str:
        return self._status.value

    @property
    def severed_nodes(self) -> frozenset[str]:
        return frozenset(self._severed_nodes)

    def sever_node(self, node_id: str) -> None:
        self._severed_nodes.add(node_id)

    def trigger_lockdown(self, reason: str) -> None:
        self._status = SystemStatus.LOCKDOWN
        self._lockdown_reason = reason

    def authorize_meatspace_override(self, provided_signature: str, challenge_nonce: str) -> bool:
        """
        Requires a physical operator HMAC-SHA256 hardware signature over challenge_nonce
        to lift LOCKDOWN and restore 25% Convex Hunt execution.
        """
        expected = hmac.new(
            self._meatspace_secret,
            challenge_nonce.encode("utf-8"),
            "sha256",
        ).hexdigest()

        if not hmac.compare_digest(provided_signature, expected):
            raise ZeroTrustSecurityError("Invalid meatspace authorization signature. Remain in LOCKDOWN.")

        self._status = SystemStatus.NOMINAL
        self._lockdown_reason = None
        self._severed_nodes.clear()
        return True

    def compute_routing_allocation(self, total_nav: float) -> CapitalAllocationDirective:
        """
        Preserves the core 75% Fortress / 25% Convex Hunt architecture:
        - NOMINAL: 75% Fortress (sub-90-day sovereign paper) + 25% Convex Hunt active.
        - LOCKDOWN: Defaults to the 75% Fortress allocation (sub-90-day sovereign paper)
          and halts all 25% Convex Hunt executions (held in frozen sovereign reserve).
        """
        if total_nav <= 0:
            raise ValueError("Total NAV must be strictly positive.")

        fortress_allocation = round(total_nav * FORTRESS_WEIGHT, 2)
        convex_bucket = round(total_nav * CONVEX_HUNT_WEIGHT, 2)

        if self._status is SystemStatus.NOMINAL:
            return CapitalAllocationDirective(
                system_status=SystemStatus.NOMINAL,
                total_nav=total_nav,
                fortress_sovereign_paper_nav=fortress_allocation,
                fortress_weight=FORTRESS_WEIGHT,
                convex_hunt_active_nav=convex_bucket,
                convex_hunt_standby_cash_nav=0.0,
                convex_hunt_execution_enabled=True,
                lockdown_reason=None,
            )

        return CapitalAllocationDirective(
            system_status=SystemStatus.LOCKDOWN,
            total_nav=total_nav,
            fortress_sovereign_paper_nav=fortress_allocation,
            fortress_weight=FORTRESS_WEIGHT,
            convex_hunt_active_nav=0.0,
            convex_hunt_standby_cash_nav=convex_bucket,
            convex_hunt_execution_enabled=False,
            lockdown_reason=self._lockdown_reason,
        )


@dataclass(frozen=True, slots=True)
class NodeFeedBundle:
    """Encapsulates an inbound Tier-1 institutional node response for concurrent ingestion."""

    node_id: str
    raw_payload: dict[str, Any] | str | bytes
    expected_sha256: str
    sts_token: EphemeralSTSToken
    mtls_handshake: MTLSConnectionHandshake


@dataclass(frozen=True, slots=True)
class ConsensusResult:
    """Verified triangulated market state emitted by the Consensus Engine."""

    sp500_price: float
    us10y_yield: float
    vix: float
    btc_usd: float
    brent_crude: float
    verified_nodes: tuple[str, ...]
    severed_nodes: tuple[str, ...]
    spoofing_incidents: list[dict[str, Any]] = field(default_factory=list)


NodeFetcherCallable = Callable[[], Coroutine[Any, Any, NodeFeedBundle]]


class TriangulatedConsensusEngine:
    """
    Phase 2: Triangulated Consensus (Anti-Spoofing Engine).
    Concurrently fetches Core Metrics from 3 independent Tier-1 institutional APIs,
    applies > 0.5 standard deviation median outlier rejection, severs spoofed feeds,
    and triggers SYSTEM_STATUS = 'LOCKDOWN' if verification quorum fails.
    """

    def __init__(
        self,
        gateway: ZeroTrustGatewayMiddleware,
        validator: DeterministicValidationEngine,
        state_manager: AFOSKernelStateManager,
        quarantine_sink: QuarantineSink,
    ) -> None:
        self._gateway = gateway
        self._validator = validator
        self._state = state_manager
        self._quarantine = quarantine_sink

    async def execute_consensus_cycle(
        self,
        tier1_fetchers: dict[str, NodeFetcherCallable],
        now_ts: float | None = None,
    ) -> ConsensusResult:
        if len(tier1_fetchers) != 3:
            raise ValueError("Triangulated Consensus strictly requires 3 independent Tier-1 feeds.")

        active_fetchers = {
            node_id: fetcher
            for node_id, fetcher in tier1_fetchers.items()
            if node_id not in self._state.severed_nodes
        }

        if len(active_fetchers) < MIN_VERIFIED_QUORUM:
            reason = (
                f"Pre-flight quorum failure: only {len(active_fetchers)} unsevered feeds available "
                f"(minimum {MIN_VERIFIED_QUORUM} required)."
            )
            self._state.trigger_lockdown(reason)
            raise ConsensusLockdownError(reason)

        # Concurrent Multi-Node Fetching
        tasks = {
            node_id: asyncio.create_task(self._ingest_and_validate_node(node_id, fetcher, now_ts))
            for node_id, fetcher in active_fetchers.items()
        }
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)

        valid_payloads: dict[str, CoreMetricsPayload] = {}
        for node_id, outcome in zip(tasks.keys(), results):
            if isinstance(outcome, Exception):
                self._state.sever_node(node_id)
            else:
                valid_payloads[node_id] = outcome

        if len(valid_payloads) < MIN_VERIFIED_QUORUM:
            reason = (
                f"Gateway/Schema quorum failure: only {len(valid_payloads)} of 3 feeds passed "
                "cryptographic and physical validation."
            )
            self._state.trigger_lockdown(reason)
            raise ConsensusLockdownError(reason)

        # Phase 2.2: Algorithmic Outlier Rejection (> 0.5 standard deviations from median)
        spoofing_incidents: list[dict[str, Any]] = []
        outlier_nodes: set[str] = set()

        if len(valid_payloads) == 3:
            for metric in METRIC_FIELDS:
                node_values = {
                    node_id: float(getattr(payload, metric))
                    for node_id, payload in valid_payloads.items()
                }
                values_list = list(node_values.values())
                median_val = statistics.median(values_list)
                raw_stdev = statistics.stdev(values_list)
                effective_stdev = max(raw_stdev, abs(median_val) * MICROSTRUCTURE_REL_NOISE_FLOOR, 1e-6)

                for node_id, val in node_values.items():
                    deviation = abs(val - median_val)
                    sigma_distance = deviation / effective_stdev

                    if sigma_distance > OUTLIER_SIGMA_THRESHOLD:
                        outlier_nodes.add(node_id)
                        incident = {
                            "event": "Adversarial_Spoofing_Attempt",
                            "node_id": node_id,
                            "metric": metric,
                            "reported_value": val,
                            "median_value": median_val,
                            "stdev": raw_stdev,
                            "sigma_distance": round(sigma_distance, 4),
                            "threshold_sigma": OUTLIER_SIGMA_THRESHOLD,
                        }
                        spoofing_incidents.append(incident)
                        await self._quarantine.dump(
                            failure_category="Adversarial_Spoofing_Attempt",
                            source_node=node_id,
                            reason=(
                                f"Outlier rejected on '{metric}': value {val} deviates from median "
                                f"{median_val} by {sigma_distance:.2f} sigma (> {OUTLIER_SIGMA_THRESHOLD} sigma)."
                            ),
                            raw_payload=valid_payloads[node_id].model_dump(),
                        )

        for rogue_node in outlier_nodes:
            self._state.sever_node(rogue_node)
            valid_payloads.pop(rogue_node, None)

        if len(valid_payloads) < MIN_VERIFIED_QUORUM:
            reason = (
                f"Triangulated consensus broken: {len(outlier_nodes)} feed(s) flagged for "
                f"Adversarial_Spoofing_Attempt, leaving {len(valid_payloads)} verified feed(s)."
            )
            self._state.trigger_lockdown(reason)
            raise ConsensusLockdownError(reason)

        # Verify remaining 2 feeds agree within tight institutional tolerance (0.5% max divergence)
        if len(valid_payloads) == 2:
            remaining_nodes = list(valid_payloads.keys())
            p_a, p_b = valid_payloads[remaining_nodes[0]], valid_payloads[remaining_nodes[1]]
            for metric in METRIC_FIELDS:
                val_a = float(getattr(p_a, metric))
                val_b = float(getattr(p_b, metric))
                denom = max(abs(val_a), abs(val_b), 1e-6)
                if abs(val_a - val_b) / denom > 0.005:
                    reason = (
                        f"Bilateral divergence on '{metric}' between remaining feeds "
                        f"{remaining_nodes[0]} ({val_a}) and {remaining_nodes[1]} ({val_b})."
                    )
                    self._state.trigger_lockdown(reason)
                    raise ConsensusLockdownError(reason)

        # Compute final consensus metrics from remaining verified feeds
        aggregated: dict[str, float] = {}
        for metric in METRIC_FIELDS:
            verified_vals = [float(getattr(p, metric)) for p in valid_payloads.values()]
            aggregated[metric] = round(statistics.median(verified_vals), 6)

        return ConsensusResult(
            sp500_price=aggregated["sp500_price"],
            us10y_yield=aggregated["us10y_yield"],
            vix=aggregated["vix"],
            btc_usd=aggregated["btc_usd"],
            brent_crude=aggregated["brent_crude"],
            verified_nodes=tuple(sorted(valid_payloads.keys())),
            severed_nodes=tuple(sorted(self._state.severed_nodes)),
            spoofing_incidents=spoofing_incidents,
        )

    async def _ingest_and_validate_node(
        self,
        node_id: str,
        fetcher: NodeFetcherCallable,
        now_ts: float | None,
    ) -> CoreMetricsPayload:
        bundle = await fetcher()
        envelope = await self._gateway.ingest_request(
            node_id=node_id,
            raw_payload=bundle.raw_payload,
            expected_sha256=bundle.expected_sha256,
            sts_token=bundle.sts_token,
            mtls_handshake=bundle.mtls_handshake,
            required_scope="ingest:metrics",
            now_ts=now_ts,
        )
        return await self._validator.validate_core_metrics(
            node_id=node_id,
            raw_data=envelope.payload,
        )
