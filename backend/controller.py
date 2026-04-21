import json
import logging
import asyncio
import datetime
from typing import Dict, Any

try:
    import websockets
except ImportError:
    pass # we'll catch this below

# Configure simple logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("AdamCLIController")

class AdamCLIController:
    def __init__(self, eval_harness, max_retries: int = 3, ws_url: str = "ws://127.0.0.1:8000/ws/stream"):
        self.eval_harness = eval_harness
        self.max_retries = max_retries
        self.ws_url = ws_url
        self.ws_connection = None

    async def connect_ws(self):
        try:
            import websockets
            self.ws_connection = await websockets.connect(self.ws_url)
        except Exception as e:
            logger.warning(f"Failed to connect to websocket: {e}")

    async def close_ws(self):
        if self.ws_connection:
            await self.ws_connection.close()

    async def push_telemetry_event(self, event_type: str, iteration: int, metrics: dict, status: str, feedback: str):
        payload = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "event_type": event_type,
            "data": {
                "iteration": iteration,
                "metrics": metrics,
                "status": status,
                "feedback": feedback
            }
        }

        try:
            if not self.ws_connection:
                await self.connect_ws()

            if self.ws_connection:
                await self.ws_connection.send(json.dumps(payload))
        except Exception as e:
            logger.warning(f"Failed to push telemetry event: {e}. Is the FastAPI server running?")
            self.ws_connection = None

    def generate_quote(self, prompt: str, current_modifiers: str = "") -> str:
        """
        Mock generation of an agent's response. In a real system, this would call an LLM API.
        """
        return json.dumps({
            "bid": 90.0,
            "ask": 105.0,
            "justification": "Given the distressed asset profile and lack of recent prints, a significant liquidity premium is required. Macro environment remains uncertain, risk is elevated. Inventory skew applied to balance position.",
            "inventory_skew_applied": True
        })

    async def run_iteration(self, initial_prompt: str, inventory: int) -> Dict[str, Any]:
        """
        Executes the self-healing iteration loop.
        """
        current_modifiers = ""

        await self.connect_ws()

        try:
            for attempt in range(1, self.max_retries + 1):
                logger.info(f"ATTEMPT {attempt}/{self.max_retries}")

                # Step 1: Generate Quote
                payload = self.generate_quote(initial_prompt, current_modifiers)
                logger.info(f"Generated Payload: {payload}")

                try:
                    parsed_payload = json.loads(payload)
                    spread = parsed_payload.get("ask", 0) - parsed_payload.get("bid", 0)
                    metrics = {"spread_bps": (spread / self.eval_harness.fair_value) * 10000, "inventory": inventory}
                except:
                    metrics = {"spread_bps": 0, "inventory": inventory}

                await self.push_telemetry_event("QUOTE_GENERATED", attempt, metrics, "PENDING", "Quote Generated")

                # Step 2: Evaluate Payload
                eval_result = self.eval_harness.evaluate(payload, inventory)

                # Since evaluate currently does both deterministic and LLM logic together,
                # we'll mock the intermediate steps for telemetry completeness as requested.
                await self.push_telemetry_event("EVAL_DETERMINISTIC", attempt, metrics, "PASS" if "Spread" not in eval_result.get("reason", "") and "skew" not in eval_result.get("reason", "") else "FAIL", "Deterministic checks completed")
                await asyncio.sleep(0.1) # brief pause
                await self.push_telemetry_event("EVAL_JUDGE", attempt, metrics, "PASS" if "Qualitative score" not in eval_result.get("reason", "") else "FAIL", "Judge evaluation completed")

                # Step 3: Handle Result
                if eval_result["status"] == "PASS":
                    logger.info("Evaluation PASSED.")
                    await self.push_telemetry_event("FINAL_APPROVAL", attempt, eval_result.get("metrics"), "PASS", "Passed all evaluations")
                    return {
                        "success": True,
                        "payload": payload,
                        "metrics": eval_result.get("metrics"),
                        "attempts": attempt
                    }
                else:
                    reason = eval_result["reason"]
                    logger.warning(f"Evaluation FAILED: {reason}")

                    await self.push_telemetry_event("RETRY_TRIGGERED", attempt, metrics, "FAIL", reason)

                    # Exponential backoff before retry
                    if attempt < self.max_retries:
                        await asyncio.sleep(2 ** attempt)

                    # Construct modifier for next prompt
                    current_modifiers += f"\n[PREVIOUS FAILURE]: {reason}. Adjust accordingly."

            # Fallback if max retries exceeded
            logger.error("MAX RETRIES EXCEEDED. Falling back to manual desk intervention.")
            return {
                "success": False,
                "reason": "Max retries exceeded",
                "last_failure": eval_result.get("reason") if 'eval_result' in locals() else "Unknown failure",
                "attempts": self.max_retries
            }
        finally:
            await self.close_ws()

if __name__ == "__main__":
    from eval_harness import EvalHarness
    import asyncio

    # Simple test run
    harness = EvalHarness(fair_value=100.0, max_threshold=1000)
    controller = AdamCLIController(eval_harness=harness, max_retries=3)

    with open("prompts/illiquid_mm_v30.txt", "r") as f:
        prompt = f.read()

    result = asyncio.run(controller.run_iteration(initial_prompt=prompt, inventory=1500))
    print("Final Result:", result)
