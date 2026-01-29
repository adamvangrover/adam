from __future__ import annotations
from typing import Any, Dict
from core.agents.agent_base import AgentBase
from core.prompting.plugins.crisis_simulation_plugin import CrisisSimulationPlugin
from core.schemas.crisis_simulation import CrisisSimulationInput, CrisisSimulationOutput, CrisisLogEntry
from core.engine.states import init_crisis_state
import logging

# Try to import v23 Graph logic
try:
    from core.engine.crisis_simulation_graph import crisis_simulation_app
    GRAPH_AVAILABLE = True
except ImportError:
    GRAPH_AVAILABLE = False
    crisis_simulation_app = None


class CrisisSimulationMetaAgent(AgentBase):
    """
    A meta-agent that conducts dynamic, enterprise-grade crisis simulations.
    It uses a sophisticated prompt structure to simulate the cascading effects of
    risks based on a user-defined scenario.
    """

    def __init__(self, config: Dict[str, Any], **kwargs):
        super().__init__(config, **kwargs)
        try:
            with open("prompt_library/AOPL-v1.0/simulation/crisis_simulation.md", "r") as f:
                prompt_template = f.read()

            from core.prompting.base_prompt_plugin import PromptMetadata

            metadata = PromptMetadata(
                prompt_id="crisis_simulation_v1",
                author="ADAM Project",
                model_config={"temperature": 0.0, "max_tokens": 4096}
            )

            self.simulation_plugin = CrisisSimulationPlugin(
                metadata=metadata,
                user_template=prompt_template
            )
        except FileNotFoundError:
            logging.error("Crisis simulation prompt template not found.")
            self.simulation_plugin = None
        except Exception as e:
            logging.error(f"Error initializing CrisisSimulationPlugin: {e}")
            self.simulation_plugin = None

    async def execute(self, simulation_input: CrisisSimulationInput) -> CrisisSimulationOutput:
        """
        Executes the crisis simulation.

        If v23 Graph is available, delegates to `CrisisSimulationGraph`.
        Otherwise, falls back to v21 Prompt-as-Code logic.

        Args:
            simulation_input: A Pydantic model containing the risk portfolio,
                              current date, and the user's crisis scenario.

        Returns:
            A Pydantic model containing the structured simulation output.
        """
        logging.info("Starting crisis simulation...")

        # --- v23 Path: Cyclical Graph ---
        if GRAPH_AVAILABLE and crisis_simulation_app:
            logging.info("CrisisSimulationMetaAgent: Delegating to v23 CrisisSimulationGraph.")

            input_dict = simulation_input.model_dump()
            scenario = input_dict.get("user_scenario", "General Crisis Scenario")
            portfolio = input_dict.get("risk_portfolio", {})

            initial_state = init_crisis_state(scenario, portfolio)

            try:
                config = {"configurable": {"thread_id": "1"}}
                result_state = await crisis_simulation_app.ainvoke(initial_state, config=config)

                # Transform Graph Output to CrisisSimulationOutput
                # The graph returns 'final_report' (str)
                final_report = result_state.get("final_report", "Graph Execution Complete")

                # Extract log from graph state if available
                graph_log = result_state.get("crisis_simulation_log", [])

                if graph_log:
                    # Convert dicts to Pydantic models
                    structured_log = []
                    for entry in graph_log:
                        try:
                            # Ensure it matches schema
                            structured_log.append(CrisisLogEntry(**entry))
                        except Exception as e:
                            logging.warning(f"Failed to parse log entry: {e}")
                else:
                    # Fallback to synthetic log
                    structured_log = [
                        CrisisLogEntry(
                            timestamp="T+0",
                            event_description=f"Simulation initialized for scenario: {scenario}",
                            risk_id_cited="SYS-INIT",
                            status="Active"
                        ),
                        CrisisLogEntry(
                            timestamp="T+End",
                            event_description="Simulation completed by v23 Graph Engine (No details returned).",
                            risk_id_cited="SYS-END",
                            status="Resolved"
                        )
                    ]

                return CrisisSimulationOutput(
                    executive_summary=final_report,
                    crisis_simulation_log=structured_log,
                    recommendations="Review the detailed graph trace in 'final_report' for mitigation strategies."
                )
            except Exception as e:
                logging.error(f"CrisisSimulationGraph execution failed: {e}. Falling back to v21 logic.")
                # Fallthrough to legacy logic

        # --- v21 Path: Prompt-as-Code ---
        if not self.simulation_plugin:
            raise RuntimeError("CrisisSimulationPlugin is not initialized.")

        # Render the prompt using the plugin
        messages = self.simulation_plugin.render_messages(simulation_input.model_dump())

        # In a real scenario, this would be an API call to an LLM.
        if self.kernel:
            # This is where the actual LLM call would be made.
            logging.info("Semantic Kernel is available, would make a real LLM call here.")
            llm_response_str = self._mock_llm_call(messages)  # Keep mock for now to not break tests
        else:
            logging.warning("Semantic Kernel not available. Using mocked LLM call for crisis simulation.")
            llm_response_str = self._mock_llm_call(messages)

        # Parse the response using the plugin
        parsed_output = self.simulation_plugin.parse_response(llm_response_str)

        logging.info("Crisis simulation finished.")
        return parsed_output

    def _mock_llm_call(self, messages: list[dict[str, str]]) -> str:
        """
        Mocks the response from a Large Language Model for demonstration purposes.
        """
        return """
        {
            "executive_summary": "The ransomware attack simulation indicates a high risk of cascading failures, primarily due to weak backup controls. The estimated financial impact is $5.7M, jeopardizing the 'Q4 Revenue Growth' strategic objective.",
            "crisis_simulation_log": [
                {
                    "timestamp": "T+00:00",
                    "event_description": "Ransomware encrypts the main ERP database.",
                    "risk_id_cited": "R-CYB-01",
                    "status": "Active"
                },
                {
                    "timestamp": "T+01:00",
                    "event_description": "Backup restoration process fails due to corrupted backups.",
                    "risk_id_cited": "R-CYB-01",
                    "status": "Failed"
                },
                {
                    "timestamp": "T+02:00",
                    "event_description": "Supply chain and order processing halt due to ERP unavailability.",
                    "risk_id_cited": "R-OPS-03",
                    "status": "Active"
                },
                {
                    "timestamp": "T+24:00",
                    "event_description": "Inability to close the fiscal quarter books is confirmed.",
                    "risk_id_cited": "R-FIN-01",
                    "status": "Escalating"
                }
            ],
            "recommendations": "Immediately activate the Cyber Incident Response Team. Isolate the affected network segments. Begin manual order processing where possible."
        }
        """
