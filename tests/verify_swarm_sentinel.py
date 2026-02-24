import asyncio
import logging
from core.engine.swarm.hive_mind import HiveMind

logging.basicConfig(level=logging.INFO)

async def run_verification():
    print("Initializing HiveMind...")
    hive = HiveMind(worker_count=2) # Minimal workers
    await hive.initialize()

    # Verify Sentinel exists
    sentinels = [w for w in hive.workers if w.role == "sentinel"]
    if not sentinels:
        print("FAILURE: No SentinelWorker spawned.")
        await hive.shutdown()
        return
    print(f"SentinelWorker spawned: {sentinels[0].id}")

    # Simulate an Analysis Result (High Risk)
    print("Simulating High Risk Analysis Result...")
    risk_payload = {"target": "DistressedCorp", "risk_score": 95}
    await hive.board.deposit("ANALYSIS_RESULT", risk_payload, intensity=10.0, source="AnalystSim")

    # Wait for Sentinel to react (it sleeps 0.5s)
    print("Waiting for Sentinel to react...")
    await asyncio.sleep(2.0)

    # Check for RISK_ALERT
    alerts = await hive.board.sniff("RISK_ALERT")
    if alerts:
        print(f"SUCCESS: Found {len(alerts)} RISK_ALERT(s).")
        print(f"Alert Data: {alerts[0].data}")
        assert alerts[0].data["target"] == "DistressedCorp"
    else:
        print("FAILURE: No RISK_ALERT found.")

    await hive.shutdown()

if __name__ == "__main__":
    asyncio.run(run_verification())
