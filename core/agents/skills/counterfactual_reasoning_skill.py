# core/agents/skills/counterfactual_reasoning_skill.py

from semantic_kernel.skill_definition import sk_function

from core.analysis.counterfactual_engine import CounterfactualEngine

# Developer Note: This skill provides an interface to the counterfactual
# reasoning capabilities of the system. Agents can use this skill to ask
# "what-if" questions, which allows for a deeper level of analysis.

class CounterfactualReasoningSkill:
    """
    A Semantic Kernel skill for counterfactual reasoning.
    """

    def __init__(self, neo4j_driver):
        self.engine = CounterfactualEngine(neo4j_driver)

    @sk_function(
        description="Answers a 'what-if' question using causal inference.",
        name="answer_what_if",
    )
    def answer_what_if(self, question: str) -> str:
        """
        Answers a "what-if" question.

        Args:
            question: The "what-if" question to answer.

        Returns:
            A string containing the answer.
        """
        return self.engine.answer_what_if(question)
