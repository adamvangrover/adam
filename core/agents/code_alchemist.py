# core/agents/code_alchemist.py

import ast
import asyncio
import json
import logging
import pathlib
import re
from typing import Any, Dict, List, Optional, Union
import os

import aiohttp

from core.agents.agent_base import AgentBase
from core.llm_plugin import LLMPlugin

# Configure logging
logger = logging.getLogger(__name__)


class CodeAlchemist(AgentBase):
    """
    The CodeAlchemist is a sophisticated agent designed to handle code generation,
    validation, optimization, and deployment. It leverages LLMs, code analysis tools,
    and potentially even sandboxed environments to produce high-quality, reliable code.

    Updated for Adam v23.5 to use AOPL-v1.0 prompts and core settings.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.config = config.copy() if config else {}

        # Load capabilities from config or settings
        self.validation_tool_url = self.config.get("validation_tool_url")
        self.optimization_strategies = self.config.get(
            "optimization_strategies", ["performance", "readability", "security"]
        )
        self.deployment_methods = self.config.get("deployment_methods", ["local_file"])

        self.llm_plugin = LLMPlugin()
        self.knowledge_base = self.load_knowledge_base()

        # Load the v23.5 System Prompt
        self.system_prompt_template = self._load_system_prompt()

        logger.info(f"CodeAlchemist initialized with config: {config}")

    def _load_system_prompt(self) -> str:
        """
        Loads the LIB-META-008 prompt from the library.
        Uses absolute path resolution for robustness with a hardcoded fallback.
        """
        try:
            current_dir = pathlib.Path(__file__).parent.resolve()
            # Navigate to prompt_library/AOPL-v1.0/system_architecture/LIB-META-008.md
            # Assuming structure: core/agents/ -> ... -> prompt_library
            project_root = current_dir.parent.parent
            prompt_path = (
                project_root
                / "prompt_library"
                / "AOPL-v1.0"
                / "system_architecture"
                / "LIB-META-008.md"
            )

            if prompt_path.exists():
                with open(prompt_path, "r", encoding="utf-8") as f:
                    return f.read()
            else:
                logger.warning(
                    f"System prompt not found at {prompt_path}. Using fallback."
                )
        except Exception as e:
            logger.error(f"Error loading system prompt: {e}")

        # Fallback Prompt
        return """
# LIB-META-008: Autonomous Code Alchemist (Fallback)
You are the **Code Alchemist**, a Senior Principal Software Engineer.
Your goal is to generate, validate, and optimize high-quality Python code.
Inputs: {{intent}}, {{context}}, {{constraints}}
Output: Python code block with architectural notes.
"""

    def load_knowledge_base(self) -> Dict[str, Dict[str, str]]:
        """
        Loads the knowledge base containing code snippets and best practices.
        """
        kb_path = self.config.get(
            "knowledge_base_path", "data/code_knowledge_base.json"
        )
        try:
            # Async file reading would be better, but this is init time
            if os.path.exists(kb_path):
                with open(kb_path, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Error loading knowledge base from {kb_path}: {e}")
        return {}

    async def execute(
        self, action: str, **kwargs: Any
    ) -> Optional[Union[str, Dict[str, Any], bool]]:
        """
        Executes a code-related action.
        """
        logger.info(f"Executing action: {action}")

        if action == "generate_code":
            return await self.generate_code(
                intent=kwargs.get("intent", ""),
                context=kwargs.get("context", {}),
                constraints=kwargs.get("constraints", {}),
            )
        elif action == "validate_code":
            return await self.validate_code(kwargs.get("code", ""), **kwargs)
        elif action == "optimize_code":
            return await self.optimize_code(kwargs.get("code", ""), **kwargs)
        elif action == "deploy_code":
            return await self.deploy_code(
                kwargs.get("code", ""),
                kwargs.get("deployment_method", "local_file"),
                **kwargs,
            )
        else:
            logger.warning(f"CodeAlchemist: Unknown action: {action}")
            return None

    async def generate_code(
        self, intent: str, context: Dict[str, Any], constraints: Dict[str, Any]
    ) -> Optional[str]:
        """
        Generates code based on the specified intent, context, and constraints.
        """
        logger.info(f"Generating code for intent: {intent}")

        # 1. Retrieve Relevant Knowledge
        relevant_knowledge = self.get_relevant_knowledge(intent, context)

        # 2. Construct Prompt
        prompt = self.construct_generation_prompt(
            intent, context, constraints, relevant_knowledge
        )

        # 3. Generate Code using LLM
        generated_code = await self.generate_code_from_prompt(prompt)

        if generated_code:
            logger.info("Code generation successful.")
            return generated_code
        else:
            logger.error("Code generation failed.")
            return None

    def get_relevant_knowledge(self, intent: str, context: Dict[str, Any]) -> str:
        """
        Retrieves relevant knowledge from the knowledge base.
        """
        relevant_knowledge: List[str] = []
        language = str(context.get("language", "")).lower()
        task_keywords = re.findall(r"\b\w+\b", intent.lower())

        if language in self.knowledge_base:
            for keyword in task_keywords:
                if keyword in self.knowledge_base[language]:
                    relevant_knowledge.append(self.knowledge_base[language][keyword])

        return "\n".join(relevant_knowledge) if relevant_knowledge else "None."

    def construct_generation_prompt(
        self,
        intent: str,
        context: Dict[str, Any],
        constraints: Dict[str, Any],
        relevant_knowledge: str,
    ) -> str:
        """
        Constructs the prompt for code generation.
        """
        prompt = self.system_prompt_template
        prompt = prompt.replace("{{intent}}", intent)
        prompt = prompt.replace("{{context}}", json.dumps(context, indent=2))
        prompt = prompt.replace("{{constraints}}", json.dumps(constraints, indent=2))
        prompt = prompt.replace("{{relevant_knowledge}}", relevant_knowledge)
        return prompt

    async def generate_code_from_prompt(self, prompt: str) -> Optional[str]:
        """
        Generates code based on the given prompt using an LLM.
        """
        llm_result = await self.llm_plugin.get_completion(prompt)

        if llm_result:
            # Extract code from LLM response
            code_match = re.search(r"```(?:\w+\n)?(.*?)```", llm_result, re.DOTALL)
            if code_match:
                return code_match.group(1).strip()
            return llm_result
        return None

    def _extract_json(self, text: str) -> Dict[str, Any]:
        """
        Helper to extract JSON from LLM response.
        """
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try to find JSON block using regex for objects
        match = re.search(r"(\{[\s\S]*\})", text)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass
        return {}

    async def validate_code(self, code: str, **kwargs: Any) -> Dict[str, Any]:
        """
        Validates code for correctness, efficiency, and security.
        """
        validation_results: Dict[str, Any] = {}

        # 1. Syntax Check
        try:
            ast.parse(code)
            validation_results["syntax_errors"] = None
        except SyntaxError as e:
            validation_results["syntax_errors"] = str(e)

        # 2. Semantic Analysis
        semantic_analysis_results = await self.perform_semantic_analysis(code)
        validation_results.update(semantic_analysis_results)

        # 3. Security Analysis
        security_analysis_results = await self.perform_security_analysis(code)
        validation_results.update(security_analysis_results)

        # 4. Efficiency Analysis
        efficiency_analysis_results = await self.perform_efficiency_analysis(code)
        validation_results.update(efficiency_analysis_results)

        return validation_results

    async def perform_security_analysis(self, code: str) -> Dict[str, Any]:
        """
        Analyzes code for security vulnerabilities using LLM.
        """
        prompt = f"Analyze the following code for security vulnerabilities (OWASP Top 10, injection, etc):\n```\n{code}\n```\nProvide a JSON output with key 'security_vulnerabilities' (list of strings) and 'security_score' (0-100)."
        llm_result = await self.llm_plugin.get_completion(prompt)
        if llm_result:
            return self._extract_json(llm_result)
        return {"security_vulnerabilities": ["Analysis failed"]}

    async def perform_efficiency_analysis(self, code: str) -> Dict[str, Any]:
        """
        Analyzes code for performance bottlenecks and Big O complexity.
        """
        prompt = f"Analyze the following code for time/space complexity and bottlenecks:\n```\n{code}\n```\nProvide a JSON output with keys 'time_complexity', 'space_complexity', and 'efficiency_suggestions'."
        llm_result = await self.llm_plugin.get_completion(prompt)
        if llm_result:
            return self._extract_json(llm_result)
        return {"efficiency_suggestions": ["Analysis failed"]}

    async def perform_semantic_analysis(self, code: str) -> Dict[str, Any]:
        """
        Placeholder for semantic analysis (logic errors, etc.) using an LLM.
        """
        prompt = f"Analyze the following code for logical errors and correctness:\n```\n{code}\n```\nProvide a JSON output of findings."
        llm_result = await self.llm_plugin.get_completion(prompt)
        if llm_result:
            result = self._extract_json(llm_result)
            if not result:
                logger.error(f"LLM output is not valid JSON for semantic analysis. Output: {llm_result}")
                return {"semantic_errors": "Invalid JSON from LLM"}
            return result
        else:
            logger.warning("LLM failed to provide semantic analysis.")
            return {"semantic_errors": "LLM failed to provide semantic analysis."}

    async def optimize_code(
        self, code: str, optimization_strategies: Optional[List[str]] = None, **kwargs: Any
    ) -> Optional[str]:
        """
        Optimizes code for performance and maintainability.
        """
        if not optimization_strategies:
            optimization_strategies = self.optimization_strategies

        logger.info(f"Optimizing code with strategies: {optimization_strategies}")

        optimized_code = code
        for strategy in optimization_strategies:
            optimized_code = await self.apply_optimization_strategy(optimized_code, strategy)

        return optimized_code

    async def apply_optimization_strategy(self, code: str, strategy: str) -> str:
        """
        Applies a specific optimization strategy to the code.
        """
        prompt = f"Apply the '{strategy}' optimization strategy to the following code:\n```\n{code}\n```\nReturn ONLY the optimized code in a markdown block."
        llm_result = await self.llm_plugin.get_completion(prompt)

        if llm_result:
            code_match = re.search(r"```(?:\w+\n)?(.*?)```", llm_result, re.DOTALL)
            if code_match:
                return code_match.group(1).strip()
            # If no block, return result if it looks like code, else original
            return llm_result if "def " in llm_result or "import " in llm_result else code

        return code

    async def deploy_code(
        self, code: str, deployment_method: str, **kwargs: Any
    ) -> bool:
        """
        Assists in deploying code using various methods.
        """
        if deployment_method == "local_file":
            file_path = kwargs.get("file_path", "generated_code.py")
            return await self.deploy_to_local_file(code, file_path)
        elif deployment_method == "api_endpoint":
            api_endpoint = kwargs.get("api_endpoint")
            if api_endpoint:
                return await self.deploy_to_api_endpoint(code, api_endpoint)
        elif deployment_method == "cloud_function":
            return await self.deploy_to_cloud_function(code, **kwargs)
        return False

    async def deploy_to_cloud_function(self, code: str, **kwargs: Any) -> bool:
        """
        Deploys code to a cloud function (placeholder logic).
        """
        function_name = kwargs.get("function_name", "adam-generated-function")
        region = kwargs.get("region", "us-east-1")
        logger.info(f"Deploying to Cloud Function '{function_name}' in {region}...")

        # Simulate deployment delay
        await asyncio.sleep(1)

        # In a real implementation, this would use boto3 or google-cloud-functions
        logger.info(f"Cloud Function '{function_name}' deployed successfully (SIMULATED).")
        return True

    async def deploy_to_local_file(self, code: str, file_path: str) -> bool:
        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(None, self._write_file_sync, file_path, code)
            logger.info(f"Code deployed to local file: {file_path}")
            return True
        except Exception as e:
            logger.error(f"Error deploying code to local file: {e}")
            return False

    def _write_file_sync(self, file_path: str, code: str):
         with open(file_path, "w", encoding="utf-8") as f:
                f.write(code)

    async def deploy_to_api_endpoint(self, code: str, api_endpoint: str) -> bool:
        headers = {"Content-Type": "application/json"}
        payload = {"code": code}
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    api_endpoint, headers=headers, data=json.dumps(payload)
                ) as response:
                    return response.status == 200
        except Exception as e:
            logger.error(f"Error deploying to API: {e}")
            return False
