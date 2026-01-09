import time
import json
import logging
from typing import Any, Dict, List, Optional, Callable
from collections import defaultdict

logger = logging.getLogger(__name__)

class ConvictionRouter:
    """
    Implements Adaptive Conviction logic as per 'The Protocol Paradox'.
    Decides between 'Direct Prompting' (System 1) and 'MCP' (System 2) based on conviction.
    """
    def __init__(self, threshold: float = 0.85):
        self.threshold = threshold

    def calculate_certainty(self, text: str, context: Dict[str, Any]) -> float:
        """
        Calculates a heuristic certainty score (0.0 to 1.0).
        In a real system, this would use logit probabilities from the LLM.
        Here we use a proxy based on text features and context completeness.
        """
        # Heuristic 1: Textual Hedging
        # If the agent's internal monologue contains hedging words, lower certainty.
        hedging_words = ["maybe", "perhaps", "might", "unsure", "unknown", "guess", "assume"]
        certainty = 0.95 # Base confidence assumed high

        lower_text = text.lower()
        match_count = 0
        for word in hedging_words:
            if word in lower_text:
                match_count += 1

        certainty -= (match_count * 0.1)

        # Heuristic 2: Missing Parameters
        # If the context explicitly flags missing parameters (e.g. from a previous validation step)
        if "missing_param" in context or "ambiguous" in context.get("status", ""):
            certainty -= 0.3

        return max(0.1, min(1.0, certainty))

    def should_use_mcp(self, task: str, context: Dict[str, Any]) -> bool:
        """
        Decides if MCP (Structured Schema) should be used based on conviction.
        Returns True if Conviction is High (Use MCP).
        Returns False if Conviction is Low (Use Direct Prompt/Elicitation).
        """
        certainty = self.calculate_certainty(task, context)
        logger.info(f"Conviction Score: {certainty:.2f} (Threshold: {self.threshold})")

        if certainty < self.threshold:
            # Low conviction -> Don't force schema. Use Elicitation (Direct Prompt)
            # "Heuristic 1: The Ambiguity Guardrail"
            return False
        return True

class StateAnchorManager:
    """
    Implements State Anchors to prevent State Drift in asynchronous workflows.
    "Heuristic 3: The Asynchronous State Anchor"
    """
    def __init__(self):
        self.anchors: Dict[str, Dict[str, Any]] = {}

    def create_anchor(self, task_id: str, state: Dict[str, Any]) -> str:
        """
        Snapshots the current state before an async operation.
        Returns an anchor_id.
        """
        anchor_id = f"anchor_{task_id}_{int(time.time())}"
        # Deep copy or serialization to ensure immutability
        try:
            self.anchors[anchor_id] = json.loads(json.dumps(state))
        except Exception as e:
            logger.warning(f"Failed to create state anchor: {e}")
            self.anchors[anchor_id] = state.copy()

        logger.info(f"Created State Anchor: {anchor_id} for task {task_id}")
        return anchor_id

    def verify_anchor(self, anchor_id: str, current_state: Dict[str, Any]) -> bool:
        """
        Verifies if the current state has drifted from the anchor.
        Returns True if state is consistent (no drift), False otherwise.
        """
        if anchor_id not in self.anchors:
            logger.warning(f"Anchor {anchor_id} not found.")
            return False

        original_state = self.anchors[anchor_id]

        # Simple drift check: Check if keys match and critical values are same
        drift_detected = False
        drift_details = []

        for key, value in original_state.items():
            if key not in current_state:
                drift_details.append(f"Key '{key}' missing")
                drift_detected = True
            elif current_state[key] != value:
                # Allow for some minor changes, but flag if critical
                drift_details.append(f"Value mismatch for '{key}'")
                drift_detected = True

        if drift_detected:
            logger.warning(f"State Drift Detected for {anchor_id}: {', '.join(drift_details)}")
            return False

        # Clean up used anchor if verification passed?
        # Usually we keep it until the task is fully done, but for now let's keep it.
        return True

class ToolRegistry:
    """
    Implements Tool RAG (Context Budgeting).
    Dynamically loads tools based on relevance to the task to prevent Context Saturation.
    "Heuristic 2: The Entropy Check"
    """
    def __init__(self, available_tools: List[Dict[str, Any]]):
        self.available_tools = available_tools # List of tool definitions (dicts)

    def retrieve_tools(self, query: str, limit: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieves the most relevant tools for the query.
        """
        if not self.available_tools:
            return []

        # Simple keyword matching for prototype
        scored_tools = []
        query_terms = set(query.lower().split())

        for tool in self.available_tools:
            score = 0
            # Check name
            tool_name = tool.get("name", "").lower()
            if tool_name in query_terms:
                score += 5

            # Check description
            tool_desc = tool.get("description", "").lower()
            for term in query_terms:
                if term in tool_desc:
                    score += 1

            if score > 0:
                scored_tools.append((score, tool))

        # Sort by score descending
        scored_tools.sort(key=lambda x: x[0], reverse=True)

        # Return top N
        results = [t[1] for t in scored_tools[:limit]]
        logger.info(f"Tool RAG retrieved {len(results)} tools for query: '{query}'")
        return results

class SubscriptionManager:
    """
    Manages event subscriptions with damping to prevent Interrupt Storms.
    Implements the "Subscription/Notification" pattern.
    """
    def __init__(self, message_broker):
        self.message_broker = message_broker
        self.last_notification_time: Dict[str, float] = defaultdict(float)
        self.damping_factor = 2.0 # Seconds between notifications for same topic

    def subscribe(self, topic: str, callback: Callable):
        """
        Subscribes to a topic via the underlying broker.
        """
        self.message_broker.subscribe(topic, callback)

    def publish_damped(self, topic: str, message: str):
        """
        Publishes a message only if enough time has passed (Damping).
        """
        current_time = time.time()
        last_time = self.last_notification_time[topic]

        if current_time - last_time > self.damping_factor:
            self.message_broker.publish(topic, message)
            self.last_notification_time[topic] = current_time
            logger.info(f"Published damped message to {topic}")
        else:
            logger.info(f"Suppressed message to {topic} (Damping active)")
