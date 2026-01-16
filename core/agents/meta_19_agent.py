# core/agents/meta_19_agent.py
from core.agents.agent_base import AgentBase
from typing import Any, Dict, List, Optional
import logging
import re


class Meta19Agent(AgentBase):
    """
    Monitors the reasoning and outputs of other agents to ensure logical consistency,
    coherence, and alignment with core principles. Deprecated as part of Adam v19 to v22.
    """

    def __init__(self, config: Dict[str, Any], **kwargs):
        super().__init__(config, **kwargs)
        # Initialization specific to this agent, e.g., loading logical fallacy patterns
        self.logical_fallacies = self.config.get("logical_fallacies", {})
        self.positive_keywords = self.config.get("positive_keywords", [])
        self.negative_keywords = self.config.get("negative_keywords", [])

        # Pre-compile regex patterns for performance
        self.compiled_fallacies = {}
        self.fallacy_prefilters = {}
        for fallacy, patterns in self.logical_fallacies.items():
            self.compiled_fallacies[fallacy] = [re.compile(p, re.IGNORECASE) for p in patterns]
            # Create a combined regex for fast pre-filtering
            if patterns:
                # Use (?:...) for non-capturing groups to ensure precedence is correct
                combined_pattern = "|".join(f"(?:{p})" for p in patterns)
                self.fallacy_prefilters[fallacy] = re.compile(combined_pattern, re.IGNORECASE)
            else:
                self.fallacy_prefilters[fallacy] = None

        # Optimize keywords by joining them into single regexes
        if self.positive_keywords:
            self.positive_pattern = re.compile("|".join(f"(?:{k})" for k in self.positive_keywords), re.IGNORECASE)
        else:
            self.positive_pattern = None

        if self.negative_keywords:
            self.negative_pattern = re.compile("|".join(f"(?:{k})" for k in self.negative_keywords), re.IGNORECASE)
        else:
            self.negative_pattern = None

        logging.info(f"MetaCognitiveAgent initialized with {len(self.logical_fallacies)} logical fallacy patterns.")

    async def execute(self, analysis_chain: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Executes the meta-cognitive analysis on a chain of agent outputs.

        Args:
            analysis_chain (List[Dict[str, Any]]): A list of dictionaries, where each
                                                  represents an agent's output in a workflow.
                                                  Example: [{"agent": "FundamentalAnalyst", "output": {...}},
                                                            {"agent": "RiskAssessment", "output": {...}}]

        Returns:
            Dict[str, Any]: A dictionary containing identified inconsistencies, fallacies,
                            and a confidence score.
        """
        logging.info("Executing meta-cognitive analysis...")
        results = {
            "inconsistencies": [],
            "logical_fallacies": [],
            "confidence_score": 1.0,  # Start with full confidence
            "summary": ""
        }

        # 1. Detect logical fallacies in the reasoning of each agent
        for step in analysis_chain:
            agent_output_text = str(step.get("output", ""))
            fallacies = self._detect_logical_fallacies(agent_output_text)
            if fallacies:
                results["logical_fallacies"].extend(fallacies)
                results["confidence_score"] -= 0.1 * len(fallacies)  # Penalize confidence

        # 2. Cross-validate outputs between different agents
        inconsistencies = self._cross_validate_outputs(analysis_chain)
        if inconsistencies:
            results["inconsistencies"].extend(inconsistencies)
            results["confidence_score"] -= 0.2 * len(inconsistencies)  # Penalize confidence more for inconsistencies

        # Ensure confidence is not below 0
        results["confidence_score"] = max(0.0, results["confidence_score"])

        # 3. Generate a summary of the meta-cognitive analysis
        results["summary"] = self._generate_summary(results)

        logging.info(f"Meta-cognitive analysis complete. Confidence score: {results['confidence_score']}.")
        return results

    def _detect_logical_fallacies(self, text: str) -> List[Dict[str, str]]:
        """
        Detects logical fallacies in a given text using regex patterns.
        """
        detected_fallacies = []
        for fallacy, patterns in self.compiled_fallacies.items():
            # Check pre-filter first to avoid iterating through all patterns if none match
            prefilter = self.fallacy_prefilters.get(fallacy)
            if prefilter and not prefilter.search(text):
                continue

            for pattern in patterns:
                if pattern.search(text):
                    detected_fallacies.append({
                        "fallacy": fallacy,
                        "evidence": f"Detected pattern '{pattern.pattern}'"
                    })
        return detected_fallacies

    def _cross_validate_outputs(self, analysis_chain: List[Dict[str, Any]]) -> List[str]:
        """
        Cross-validates outputs from different agents to find inconsistencies.
        """
        inconsistencies = []
        outputs_text = " ".join([str(step.get("output", "")) for step in analysis_chain]).lower()

        has_positive = bool(self.positive_pattern.search(outputs_text)) if self.positive_pattern else False
        has_negative = bool(self.negative_pattern.search(outputs_text)) if self.negative_pattern else False

        if has_positive and has_negative:
            inconsistencies.append(
                "Inconsistency detected: The analysis contains conflicting positive and negative statements.")

        return inconsistencies

    def _generate_summary(self, analysis_results: Dict[str, Any]) -> str:
        """
        Generates a summary of the meta-cognitive findings.
        """
        if analysis_results["confidence_score"] > 0.9:
            return "The analysis appears logically consistent and coherent."

        summary_text = f"Meta-cognitive review raised some concerns (Confidence: {analysis_results['confidence_score']:.2f}):\n"
        if analysis_results["inconsistencies"]:
            summary_text += "- Inconsistencies: " + " ".join(analysis_results["inconsistencies"]) + "\n"
        if analysis_results["logical_fallacies"]:
            summary_text += "- Potential logical fallacies detected: "
            summary_text += ", ".join(list(set([f["fallacy"] for f in analysis_results["logical_fallacies"]]))) + ".\n"

        return summary_text
