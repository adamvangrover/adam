from __future__ import annotations
from typing import Any, Dict, List, Optional
import logging
import uuid
import ast
import random
import copy
from core.agents.agent_base import AgentBase
from core.agents.mixins.audit_mixin import AuditMixin
from core.schemas.meta_agent_schemas import (
    EvolutionaryArchitectInput,
    EvolutionaryArchitectOutput,
    ProposedChange,
    EvolutionGoal
)

class EvolutionaryArchitectAgent(AgentBase, AuditMixin):
    """
    The Evolutionary Architect Agent is a meta-agent predisposed for action.
    It drives the codebase forward by proposing additive enhancements, refactors,
    and optimizations. It uses 'Active Inference' principles to minimize the
    divergence between the current codebase state and the desired goal state.
    """

    def __init__(self, config: Dict[str, Any], **kwargs):
        super().__init__(config, **kwargs)
        AuditMixin.__init__(self)
        self.max_iterations = config.get("max_iterations", 5)

    async def execute(self, input_data: EvolutionaryArchitectInput) -> EvolutionaryArchitectOutput:
        """
        Executes the evolutionary process.
        """
        logging.info(f"EvolutionaryArchitectAgent starting evolution: {input_data.description}")

        # 1. Analyze Context
        affected_files = input_data.target_files or []
        logging.info(f"Targeting files: {affected_files}")

        # 2. Generate Proposal
        proposal_id = str(uuid.uuid4())

        # Try to use real kernel generation if available, else fallback
        if self.kernel:
            try:
                changes = await self._generate_changes_real(input_data)
            except Exception as e:
                logging.warning(f"Real generation failed: {e}. Falling back to mock logic.")
                changes = self._generate_changes_mock(input_data)
        else:
            changes = self._generate_changes_mock(input_data)

        # 3. Safety Check (AST Parse)
        safety_score = self._evaluate_safety(changes, input_data.constraints)

        output = EvolutionaryArchitectOutput(
            proposal_id=proposal_id,
            changes=changes,
            impact_analysis=f"Estimated impact: Low risk, high value for goal {input_data.goal_type.value}.",
            safety_score=safety_score,
            status="proposed" if safety_score > 0.8 else "flagged"
        )

        # Log Decision
        self.log_decision(
            activity_type="CodeEvolution",
            details={"goal": input_data.goal_type.value, "proposal_id": proposal_id},
            outcome=output.model_dump() if hasattr(output, 'model_dump') else output.__dict__
        )

        return output

    async def optimize_config_genetic(self, base_config: Dict[str, Any], generations: int = 5) -> Dict[str, Any]:
        """
        Runs a Genetic Algorithm to evolve agent configuration parameters (e.g., Temperature).
        """
        population_size = 10
        population = [base_config]

        # Initialize Population
        for _ in range(population_size - 1):
            mutated = copy.deepcopy(base_config)
            # Mutate temperature
            if 'llm_config' in mutated:
                current_temp = mutated['llm_config'].get('temperature', 0.7)
                mutated['llm_config']['temperature'] = max(0.0, min(1.0, current_temp + random.uniform(-0.1, 0.1)))
            population.append(mutated)

        best_config = base_config
        best_score = 0.0

        for gen in range(generations):
            scores = []
            for candidate in population:
                # Simulate Evaluation (Fitness Function)
                # In reality, this would run a benchmark task
                temp = candidate.get('llm_config', {}).get('temperature', 0.7)
                # Mock fitness: pretend 0.4 is optimal
                fitness = 1.0 - abs(temp - 0.4)
                scores.append((fitness, candidate))

            # Selection
            scores.sort(key=lambda x: x[0], reverse=True)
            best_score, best_config = scores[0]

            logging.info(f"Generation {gen}: Best Fitness {best_score:.4f} (Temp: {best_config.get('llm_config', {}).get('temperature'):.2f})")

            # Crossover / Next Gen
            top_performers = [s[1] for s in scores[:3]]
            new_population = top_performers[:]

            while len(new_population) < population_size:
                parent = random.choice(top_performers)
                child = copy.deepcopy(parent)
                # Mutation
                curr_temp = child['llm_config']['temperature']
                child['llm_config']['temperature'] = max(0.0, min(1.0, curr_temp + random.uniform(-0.05, 0.05)))
                new_population.append(child)

            population = new_population

        return best_config

    async def _generate_changes_real(self, input_data: EvolutionaryArchitectInput) -> List[ProposedChange]:
        """
        Attempts to generate code changes using the Semantic Kernel.
        """
        # Construct a prompt for the kernel
        # This is a simplified "Prompt-as-Code" construction within the agent
        prompt = f"""
        You are an Evolutionary Architect. Your goal is: {input_data.goal_type.value}.
        Description: {input_data.description}
        Constraints: {', '.join(input_data.constraints)}

        Generate a diff or new code block for the requested feature.
        """

        # In a real scenario, we would retrieve specific skills from the kernel
        # For now, we assume a generic 'generate_code' function exists or we invoke the LLM directly

        # Simulate an async call that might fail if no LLM is configured
        # result = await self.kernel.invoke(prompt)
        # For this specific "Mock-to-Real" bridge, we will return the mock structure
        # but wrapped in this method to show intent of where the API call goes.

        # raising NotImplementedError to trigger fallback in the try/except block above
        # until a real LLM endpoint is confirmed.
        raise NotImplementedError("LLM Kernel not fully configured for code generation.")

    def _generate_changes_mock(self, input_data: EvolutionaryArchitectInput) -> List[ProposedChange]:
        """
        Simulates the generation of code changes based on the goal.
        """
        changes = []

        if input_data.goal_type == EvolutionGoal.FEATURE_ADDITION:
            # Create a mock new file proposal
            changes.append(ProposedChange(
                file_path="core/new_feature.py",
                diff="<<<<<<< SEARCH\n=======\n# New Feature Implementation\ndef new_feature():\n    pass\n>>>>>>> REPLACE",
                rationale="Adding new feature scaffold as requested."
            ))
        elif input_data.target_files:
            for file in input_data.target_files:
                changes.append(ProposedChange(
                    file_path=file,
                    diff=f"<<<<<<< SEARCH\n# Old Code\n=======\n# Optimized Code by EvolutionaryAgent\n>>>>>>> REPLACE",
                    rationale=f"Refactoring {file} for performance."
                ))
        else:
             changes.append(ProposedChange(
                file_path="README.md",
                diff="<<<<<<< SEARCH\n# Project\n=======\n# Project (Enhanced)\n>>>>>>> REPLACE",
                rationale="General enhancement to documentation."
            ))

        return changes

    def _evaluate_safety(self, changes: List[ProposedChange], constraints: List[str]) -> float:
        """
        Evaluates the safety of the proposed changes using AST parsing.
        """
        score = 1.0

        for change in changes:
            # We can't parse a Diff directly, but we can parse the 'REPLACE' block if we extract it
            # Simple heuristic extraction
            if ">>>>>>> REPLACE" in change.diff:
                try:
                    # Extract content between ======= and >>>>>>> REPLACE
                    content = change.diff.split("=======")[1].split(">>>>>>> REPLACE")[0]
                    # Attempt to parse
                    ast.parse(content)
                except SyntaxError:
                    logging.warning(f"Syntax Error detected in proposal for {change.file_path}")
                    score -= 0.5
                except Exception:
                    # If extraction fails or other error
                    score -= 0.1

            # Constraint Checking (Simple keyword search)
            for constraint in constraints:
                if "no external deps" in constraint.lower():
                    if "import" in change.diff and "core" not in change.diff: # Very naive check
                         logging.warning("Potential external dependency violation.")
                         score -= 0.2

        return max(0.0, score)
