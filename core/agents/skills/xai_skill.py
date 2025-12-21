# core/agents/skills/xai_skill.py

from semantic_kernel.skill_definition import sk_function, sk_description
from core.analysis.xai.explainer import Explainer

# Developer Note: This skill provides an interface to the XAI capabilities
# of the system. Agents can use this skill to request explanations for
# activities, which helps to build a more transparent and auditable system.


class XAISkill:
    """
    A Semantic Kernel skill for eXplainable AI.
    """

    def __init__(self, neo4j_driver, llm_kernel):
        self.explainer = Explainer(neo4j_driver, llm_kernel)

    @sk_function(
        description="Generates a human-readable explanation for a given activity ID.",
        name="explain_activity",
    )
    def explain_activity(self, activity_id: str) -> str:
        """
        Generates an explanation for a given activity ID.

        Args:
            activity_id: The ID of the activity to explain.

        Returns:
            A string containing the explanation.
        """
        return self.explainer.explain_activity(activity_id)
