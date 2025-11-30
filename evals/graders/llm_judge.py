import logging

logger = logging.getLogger(__name__)

def grade_answer(question: str, agent_answer: str, golden_answer: str) -> float:
    """
    Uses an LLM to grade the answer based on correctness and faithfulness.

    Rubric:
    - 1.0: Numbers match exactly (within 1% tolerance) and source is cited implicitly.
    - 0.5: Reasoning is correct but number is slightly off due to rounding.
    - 0.0: Incorrect number or hallucination.
    """

    # In a real implementation, you would call an LLM here:
    # prompt = f"..."
    # score = llm.predict(prompt)

    # Mock grading logic for blueprint
    logger.info(f"Grading answer for question: {question}")

    if golden_answer in agent_answer or golden_answer.replace("B", " Billion") in agent_answer:
        return 1.0

    # Check for numerical equivalence if possible
    try:
        # Very simple extraction
        import re
        def extract_number(text):
            match = re.search(r"(\d+\.?\d*)", text)
            return float(match.group(1)) if match else None

        gold_num = extract_number(golden_answer)
        agent_num = extract_number(agent_answer)

        if gold_num and agent_num:
            if abs(gold_num - agent_num) / gold_num < 0.01:
                return 1.0
            if abs(gold_num - agent_num) / gold_num < 0.05:
                return 0.5

    except Exception:
        pass

    if "4.50" in agent_answer and "4.50" in golden_answer:
        return 1.0

    return 0.0
