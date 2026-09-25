# ADAM Dual-Plane Architecture: Build Plan & Evaluation Prompt

## 1. Goal
Operationalize the Dual-Plane Architecture, separating 1L deterministic execution from 2L probabilistic/challenger governance via an event-driven telemetry bus.

## 2. Build Plan
### Phase 1: Telemetry Backbone
- Deploy Apache Kafka for low-latency telemetry transport.
- Implement zero-copy data buffer (Apache Arrow) extraction on the 1L to emit `PricingEvent` and `MarketSnapshot` payloads asynchronously.

### Phase 2: 1L Data Plane
- Deploy Rust/C++ execution kernels for hazard rate calibration and CDS pricing.
- Integrate pre-trade limit utilization checks directly into the pricing loop.

### Phase 3: 2L Control Plane
- Deploy Python-based analytics and stochastic challenger models as microservices.
- Implement jsonLogic rule engines to evaluate Divergence Analytics (Bidirectional Disparity).
- Integrate PROV-O lineage logging for auditability.

### Phase 4: Phased Rollout
- **Shadow Mode:** 2L runs silently on 1L telemetry.
- **Passive Challenger:** MRM uses 2L for reporting.
- **Active Control:** 2L triggers automated circuit breakers based on SR 11-7 thresholds.
