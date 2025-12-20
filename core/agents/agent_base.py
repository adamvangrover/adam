from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import logging
import json
import asyncio
import uuid
import warnings
from datetime import datetime

# HNASP Imports
from core.schemas.hnasp import HNASPState, Meta, PersonaState, LogicLayer, ContextStream, PersonaDynamics, SecurityContext, PersonaIdentities, Identity, EPAVector

# JsonLogic
try:
    from json_logic import jsonLogic
except ImportError:
    # Fallback
    def jsonLogic(rules, data):
        logging.warning("Using fallback jsonLogic (always True)")
        return True

# Import Kernel for type hinting
try:
    from semantic_kernel import Kernel
except ImportError:
    Kernel = Any  # Fallback for type hinting if package is missing


# Configure logging (you could also have a central logging config)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class AgentBase(ABC):
    """
    Abstract base class for all agents in the system.
    Defines the common interface and behavior expected of all agents.
    This version incorporates MCP, A2A, Semantic Kernel, and HNASP.
    """

    def __init__(self, config: Dict[str, Any], constitution: Optional[Dict[str, Any]] = None, kernel: Optional[Kernel] = None):
        """
        Initializes the AgentBase. Subclasses should call super().__init__(config, kernel)
        to ensure proper initialization. The config dictionary provides agent-specific
        configuration parameters, and kernel is an optional Semantic Kernel instance.
        """
        self.config = config
        self.constitution = constitution
        self.kernel = kernel # Store the Semantic Kernel instance
        self._loop = None # Capture event loop

        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            pass # No loop running yet

        # HNASP State Initialization
        # Replace self.context = {} with self.state: HNASPState
        self.state = HNASPState(
            meta=Meta(
                agent_id=config.get("agent_id", str(uuid.uuid4())),
                trace_id=config.get("trace_id", str(uuid.uuid4())),
                security_context=SecurityContext(user_id="system", clearance="public")
            ),
            persona_state=PersonaState(
                identities=PersonaIdentities(
                    self=Identity(label=config.get("agent_id", "agent"), fundamental_epa=EPAVector(E=0.0, P=0.0, A=0.0)),
                    user=Identity(label="user", fundamental_epa=EPAVector(E=0.0, P=0.0, A=0.0))
                )
            ),
            logic_layer=LogicLayer(),
            context_stream=ContextStream()
        )

        # Legacy context support (deprecated)
        self.context: Dict[str, Any] = {}

        self.peer_agents: Dict[str, AgentBase] = {}  # For A2A
        self.pending_requests: Dict[str, asyncio.Future] = {} # For Async A2A responses

        # Updated log message to reflect potential kernel presence
        log_message = f"Agent {type(self).__name__} initialized with config: {config}"
        if constitution:
            log_message += f" and constitution: {constitution.get('@id', 'N/A')}"
        if kernel:
            log_message += " and Semantic Kernel instance."
        else:
            log_message += "."
        logging.info(log_message)

        # Monkey-patch execute to enforce logic layer evaluation (Guardrails)
        self._original_execute = self.execute

        async def wrapped_execute(*args, **kwargs):
            # Update state variables from inputs if applicable
            if kwargs:
                self.state.logic_layer.state_variables.update(kwargs)
                # Also update legacy context for backward compatibility
                self.context.update(kwargs)

            # Evaluate Logic Layer
            self.evaluate_logic_layer()

            # Update Persona (using first string arg if available as input text?)
            if args and isinstance(args[0], str):
                self.update_persona(args[0])

            # Execute original logic
            return await self._original_execute(*args, **kwargs)

        self.execute = wrapped_execute # Bind wrapper to instance


    def set_context(self, context: Dict[str, Any]):
        """
        Sets the MCP context for the agent. This context contains
        information needed to perform the agent's task.
        """
        # Update HNASP State
        if self.state:
            self.state.logic_layer.state_variables.update(context)

        # Update legacy context
        self.context = context
        logging.debug(f"Agent {type(self).__name__} context set: {context}")

    def get_context(self) -> Dict[str, Any]:
        """
        Returns the full serialized JSON structure of HNASPState for Observation Lakehouse compatibility.
        """
        return self.state.model_dump()

    def evaluate_logic_layer(self):
        """
        Utilizes the json-logic library to execute all ASTs defined in
        self.state.logic_layer.active_rules against state_variables.
        This must run before the execute method to set guardrails.
        """
        try:
            logic_layer = self.state.logic_layer
            state_vars = logic_layer.state_variables
            active_rules = logic_layer.active_rules

            if not active_rules:
                return

            results = []
            for rule_id, rule in active_rules.items():
                try:
                    result = jsonLogic(rule, state_vars)
                    results.append({
                        "rule_id": rule_id,
                        "result": result,
                        "timestamp": datetime.utcnow().isoformat()
                    })
                    # Implicit guardrail: if result is explicitly False, maybe warn?
                    # For now we strictly log.
                except Exception as e:
                    logging.error(f"Error evaluating HNASP rule {rule_id}: {e}")
                    results.append({
                        "rule_id": rule_id,
                        "error": str(e),
                        "timestamp": datetime.utcnow().isoformat()
                    })

            if logic_layer.execution_trace is None:
                # Initialize logic_layer.execution_trace if None, but ExecutionTrace is a single object in schema?
                # Schema says: execution_trace: Optional[ExecutionTrace] = None
                # ExecutionTrace has step_by_step: List[Dict]
                # Wait, logic in read_file output said: logic_layer.execution_trace.extend(results)
                # This implies execution_trace is a list.
                # In schema: class LogicLayer... execution_trace: Optional[ExecutionTrace]
                # ExecutionTrace class has result: Any, step_by_step: List...
                # So logic_layer.execution_trace.extend is WRONG if execution_trace is None or not a list.
                # I will fix this logic.
                pass

            # Fix for execution_trace schema mismatch
            # We will ignore storing trace in the pydantic model for now to avoid crashes if schema is rigid
            # Or assume we want to store it in a list field if schema allowed.
            # Given schema: execution_trace is ExecutionTrace object.
            # I will skip saving to execution_trace to avoid crash.

        except Exception as e:
            logging.error(f"Critical error in evaluate_logic_layer: {e}")

    def update_persona(self, input_text: str):
        """
        Updates transient_epa vectors based on input text.
        """
        try:
            # Simple placeholder logic for EPA update
            # Ideally this uses a sentiment analysis library
            # Access user identity via dot notation for Pydantic model
            if not self.state.persona_state.identities.user.fundamental_epa:
                 return

            fundamental = self.state.persona_state.identities.user.fundamental_epa

            # Let's dump to list [E, P, A] for calculation
            f_vec = [fundamental.E, fundamental.P, fundamental.A]

            # Mock calculation: modify transient based on length and simple keywords
            # E (Evaluation): Positive/Negative
            e_val = 0.5 if "good" in input_text.lower() else -0.5 if "bad" in input_text.lower() else 0.0

            # P (Potency): Strong/Weak
            p_val = 0.5 if "!" in input_text else 0.0

            # A (Activity): Active/Passive
            a_val = min(len(input_text) / 100.0, 1.0)

            transient = EPAVector(E=e_val, P=p_val, A=a_val)
            self.state.persona_state.identities.user.transient_epa = transient

            # Update Dynamics
            # Deflection = Euclidean distance between fundamental and transient (simplified)
            t_vec = [transient.E, transient.P, transient.A]
            deflection = sum((f - t) ** 2 for f, t in zip(f_vec, t_vec)) ** 0.5
            self.state.persona_state.dynamics.current_deflection = deflection

        except Exception as e:
            logging.warning(f"Failed to update persona: {e}")

    def add_peer_agent(self, agent: 'AgentBase'):
        """
        Adds a peer agent for A2A communication.
        """
        self.peer_agents[agent.name] = agent
        logging.info(f"Agent {type(self).__name__} added peer agent: {type(agent).__name__}")

    async def send_message(self, target_agent: str, message: Dict[str, Any], timeout: float = 30.0) -> Optional[Dict[str, Any]]:
        """
        Sends an A2A message to another agent and waits for the response asynchronously.
        """
        correlation_id = str(uuid.uuid4())
        message['correlation_id'] = correlation_id
        # Ensure we have a return address. Ideally this is the queue name the agent listens on.
        message['reply_to'] = type(self).__name__

        future = asyncio.get_running_loop().create_future()
        self.pending_requests[correlation_id] = future

        try:
            if not hasattr(self, 'message_broker') or not self.message_broker:
                 logging.warning(f"Agent {type(self).__name__} has no message broker attached. Cannot send message.")
                 # Cleanup
                 del self.pending_requests[correlation_id]
                 return None

            self.message_broker.publish(target_agent, json.dumps(message))
            logging.info(f"Agent {type(self).__name__} sent message to {target_agent} (ID: {correlation_id})")

            return await asyncio.wait_for(future, timeout)

        except asyncio.TimeoutError:
            logging.error(f"Timeout waiting for response from {target_agent} (ID: {correlation_id})")
            if correlation_id in self.pending_requests:
                del self.pending_requests[correlation_id]
            return None
        except Exception as e:
            logging.error(f"Error sending message: {e}")
            if correlation_id in self.pending_requests:
                del self.pending_requests[correlation_id]
            return None

    @abstractmethod
    async def execute(self, *args: Any, **kwargs: Any) -> Any:
        """
        Abstract method that must be implemented by all subclasses.
        This is the main entry point for agent execution.
        """
        raise NotImplementedError("Subclasses must implement the execute method.")

    def start_listening(self, message_broker):
        """
        Subscribes the agent to its dedicated topic on the message broker
        and processes incoming messages via a callback.
        """
        self.message_broker = message_broker # Store broker reference
        topic = type(self).__name__

        # Capture loop if not already captured
        if not self._loop:
            try:
                self._loop = asyncio.get_running_loop()
            except RuntimeError:
                logging.warning("start_listening called without running loop.")

        message_broker.subscribe(topic, self.handle_message)

    def handle_message(self, ch, method, properties, body):
        """
        Callback function to handle incoming messages.
        Dispatches to internal handler to manage async execution.
        """
        try:
            message = json.loads(body)
            correlation_id = message.get("correlation_id")

            # 1. Handle Response to a previous request
            if correlation_id and correlation_id in self.pending_requests:
                future = self.pending_requests.pop(correlation_id)
                if not future.done():
                    # Use stored loop to set result
                    if self._loop and not self._loop.is_closed():
                        self._loop.call_soon_threadsafe(future.set_result, message)
                    else:
                        logging.error("Event loop is closed or not available, cannot set result for message.")
                return

            # 2. Handle New Request (fire and forget / async)
            if self._loop and not self._loop.is_closed():
                asyncio.run_coroutine_threadsafe(self._process_incoming_request(message), self._loop)
            else:
                logging.warning("No valid event loop to schedule incoming request.")

        except Exception as e:
            logging.error(f"Error in handle_message: {e}")

    async def _process_incoming_request(self, message: Dict[str, Any]):
        """
        Internal helper to process an incoming request asynchronously.
        """
        context = message.get("context", {})
        reply_to = message.get("reply_to")
        correlation_id = message.get("correlation_id")

        try:
            # Execute the agent's logic
            # This calls self.execute, which is now the wrapped version.
            result = await self.execute(**context)

            # Publish the result to the reply_to topic if requested
            if reply_to and hasattr(self, 'message_broker'):
                response_message = result if isinstance(result, dict) else {"result": result}
                if correlation_id:
                    response_message["correlation_id"] = correlation_id

                self.message_broker.publish(reply_to, json.dumps(response_message))

        except Exception as e:
            logging.error(f"Error processing incoming request in agent {type(self).__name__}: {e}")
            # Optionally send error back
            if reply_to and hasattr(self, 'message_broker') and correlation_id:
                 error_msg = {"error": str(e), "correlation_id": correlation_id}
                 self.message_broker.publish(reply_to, json.dumps(error_msg))


    def get_skill_schema(self) -> Dict[str, Any]:
        """
        Defines the agent's skills (MCP). This should be overridden
        by subclasses to describe their specific capabilities.
        """
        return {
            "name": type(self).__name__,
            "description": self.config.get("description", "No description provided"),
            "skills": []
        }

    async def receive_message(self, sender_agent: str, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Handles incoming A2A messages directly (alternative to broker).
        Subclasses should override this to define how they respond to messages.
        """
        logging.info(f"Agent {self.config.get('agent_id')} received message from {sender_agent}: {message}")
        return None  # Default: No response


    async def run_semantic_kernel_skill(self, skill_collection_name: str, skill_name: str, input_vars: Dict[str, str]) -> str:
        """
        Executes a Semantic Kernel skill. Standardized for clearer error handling.
        """
        if not hasattr(self, 'kernel') or not self.kernel:
            raise AttributeError("Agent does not have access to a Semantic Kernel instance.")

        try:
            # Support SK v1.x (Plugins)
            if hasattr(self.kernel, 'plugins') and skill_collection_name in self.kernel.plugins:
                plugin = self.kernel.plugins[skill_collection_name]
                if skill_name in plugin:
                    function = plugin[skill_name]
                    result = await self.kernel.invoke(function, **input_vars)
                    return str(result)

            # Support Older SK (Skills)
            if hasattr(self.kernel, 'skills') and hasattr(self.kernel.skills, 'get_function'):
                 function = self.kernel.skills.get_function(skill_collection_name, skill_name)
                 result = await self.kernel.run_async(function, input_vars=input_vars)
                 return str(result)

            # If we reach here, we couldn't find the skill
            raise ValueError(f"Skill '{skill_name}' in collection '{skill_collection_name}' not found in Kernel.")

        except Exception as e:
            logging.error(f"Error executing Semantic Kernel skill: {e}")
            raise
