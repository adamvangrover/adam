# core/agents/code_alchemist.py

import os
import ast
from langchain.utilities import GoogleSearchAPIWrapper
from typing import Dict, List, Union
import re
from io import StringIO
import sys
import contextlib
import logging

# ... (import other necessary libraries)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@contextlib.contextmanager
def stdoutIO(stdout=None):
    old = sys.stdout
    if stdout is None:
        stdout = StringIO()
    sys.stdout = stdout
    yield stdout
    sys.stdout = old

class CodeAlchemist:
    def __init__(self, config):
        self.config = config
        self.search_api = GoogleSearchAPIWrapper()
        # Initialize other necessary components like the knowledge base connector
        self.knowledge_base = self._init_knowledge_base()  # Initialize the knowledge base

    def _init_knowledge_base(self):
        # Replace with your actual knowledge base initialization logic
        # This could involve connecting to a database or loading data from files
        logging.info("Initializing knowledge base...")
        # Example:
        knowledge_base = {
            "python": {
                "fibonacci": """
def fibonacci(n):
  if n <= 1:
    return n
  else:
    return(fibonacci(n-1) + fibonacci(n-2))"""
            }
        }
        logging.info("Knowledge base initialized.")
        return knowledge_base

    def generate_code(self, intent: str, context: Dict, constraints: Dict, **kwargs) -> str:
        """
        Generates code based on the specified intent, context, and constraints.

        Args:
            intent: The intended purpose or functionality of the code.
            context: Relevant context information, such as data sources or agent interactions.
            constraints: Constraints or limitations on the code, such as performance requirements or security considerations.
            **kwargs: Additional parameters for code generation.

        Returns:
            The generated code.
        """
        logging.info(f"Generating code for intent: {intent}, context: {context}, constraints: {constraints}")

        # Access relevant knowledge base modules and use them in prompt construction
        relevant_knowledge = self._get_relevant_knowledge(intent, context)

        prompt = f"""
        I am Code Alchemist, a code generation expert.
        I need to generate code that fulfills the following intent: {intent}
        Here is the relevant context: {context}
        The code should adhere to these constraints: {constraints}
        Here is some relevant knowledge you might find useful: {relevant_knowledge}

        Generate the code and explain your reasoning.
        """

        # Use the LLM to generate code based on the prompt
        generated_code = self._generate_code_from_prompt(prompt)

        logging.info(f"Generated code: {generated_code}")
        return generated_code

    def _get_relevant_knowledge(self, intent: str, context: Dict) -> str:
        """
        Retrieves relevant knowledge from the knowledge base based on intent and context.

        Args:
            intent: The intent of the code generation request.
            context: The context of the code generation request.

        Returns:
            Relevant knowledge as a string.
        """
        logging.info("Retrieving relevant knowledge from knowledge base...")
        # Example:
        if 'language' in context and context['language'] == 'python' and 'fibonacci' in intent.lower():
            return self.knowledge_base['python']['fibonacci']
        else:
            return "No relevant knowledge found."

    def _generate_code_from_prompt(self, prompt: str) -> str:
        """
        Generates code based on the given prompt.

        Args:
            prompt: The prompt for code generation.

        Returns:
            The generated code.
        """
        logging.info("Generating code from prompt...")
        # Interact with the LLM (e.g., through LangChain)
        # Use advanced language modeling techniques to generate code
        # Incorporate knowledge from the knowledge base

        # Example using LangChain (replace with your actual LLM interaction)
        from langchain.chat_models import ChatOpenAI
        from langchain import PromptTemplate, LLMChain

        llm = ChatOpenAI(temperature=0)
        template = PromptTemplate(
            input_variables=["prompt"],
            template=prompt,
        )
        chain = LLMChain(llm=llm, prompt=template)
        generated_code = chain.run(prompt)

        return generated_code

    def validate_code(self, code: str, **kwargs) -> Dict:
        """
        Validates code for correctness, efficiency, and security.

        Args:
            code: The code to validate.
            **kwargs: Additional parameters for code validation.

        Returns:
            A dictionary containing validation results and potential issues.
        """
        logging.info(f"Validating code: {code}")
        validation_results = {}

        try:
            # Check for syntax errors by parsing the code
            ast.parse(code)
            validation_results['errors'] = None
        except SyntaxError as e:
            validation_results['errors'] = f"SyntaxError: {e}"

        # TODO: Implement more comprehensive validation checks
        # - Logic errors
        # - Security vulnerabilities
        # - Use static analysis tools
        # - Execute code in a sandboxed environment

        validation_results['warnings'] =# Add any warnings
        validation_results['suggestions'] =# Add any suggestions

        logging.info(f"Validation results: {validation_results}")
        return validation_results

    def optimize_code(self, code: str, **kwargs) -> str:
        """
        Optimizes code for performance and maintainability.

        Args:
            code: The code to optimize.
            **kwargs: Additional parameters for code optimization.

        Returns:
            The optimized code.
        """
        logging.info(f"Optimizing code: {code}")
        optimized_code = code

        # TODO: Implement more advanced optimization techniques
        # - Refactoring
        # - Simplification
        # - Use code analysis tools
        # - Interact with code optimization libraries

        # Example optimization (replace with more advanced techniques)
        # Remove unnecessary whitespace and comments
        optimized_code = re.sub(r'\s+', ' ', optimized_code)
        optimized_code = re.sub(r'#.*', '', optimized_code)

        logging.info(f"Optimized code: {optimized_code}")
        return optimized_code

    def deploy_code(self, code: str, environment: str, **kwargs) -> bool:
        """
        Assists in deploying code to various environments.

        Args:
            code: The code to deploy.
            environment: The target deployment environment (e.g., local, server, cloud).
            **kwargs: Additional parameters for code deployment.

        Returns:
            True if the deployment was successful, False otherwise.
        """
        logging.info(f"Deploying code to environment: {environment}")
        # Simulate deployment based on environment
        if environment == 'local':
            # Simulate local deployment (e.g., saving to a file)
            try:
                with open('deployed_code.py', 'w') as f:
                    f.write(code)
                logging.info("Code deployed successfully to local environment.")
                return True
            except Exception as e:
                logging.error(f"Error deploying code to local environment: {e}")
                return False
        elif environment == 'server':
            # Simulate server deployment (e.g., SSH connection and file transfer)
            # ...
            logging.info("Code deployed successfully to server environment.")
            return True
        elif environment == 'cloud':
            # Simulate cloud deployment (e.g., interaction with AWS or GCP APIs)
            # ...
            logging.info("Code deployed successfully to cloud environment.")
            return True
        else:
            logging.error(f"Unsupported environment: {environment}")
            return False

    async def run(self):
        """
        Fetch code generation or optimization requests and process them.
        """
        # Fetch code generation or optimization requests from a queue or API
        # Generate, validate, optimize, or deploy code as requested
        # Example:
        request = self._fetch_request()  # Replace with your actual request fetching logic
        if request['type'] == 'generate':
            code = self.generate_code(**request['parameters'])
        elif request['type'] == 'validate':
            results = self.validate_code(**request['parameters'])
        elif request['type'] == 'optimize':
            optimized_code = self.optimize_code(**request['parameters'])
        elif request['type'] == 'deploy':
            success = self.deploy_code(**request['parameters'])

    def _fetch_request(self) -> Dict:
        """
        Fetch a code generation or optimization request.

        Returns:
            A dictionary containing the request type and parameters.
        """
        # Replace with your actual request fetching logic
        # This could involve interacting with a message queue or API
        request = {
            'type': 'generate',
            'parameters': {
                'intent': 'Generate a function to calculate the Fibonacci sequence',
                'context': {'language': 'python'},
                'constraints': {'performance': 'optimized'}
            }
        }
        return request

# Example usage
if __name__ == "__main__":
    config = {}  # Load configuration
    code_alchemist = CodeAlchemist(config)
    # asyncio.run(code_alchemist.run())  # Use asyncio.run if you have asynchronous tasks
    with stdoutIO() as s:
        code_alchemist.run()

    print(s.getvalue())





# core/agents/code_alchemist.py

import os
import ast
from typing import Any, Dict, List, Optional, Union
import re
from io import StringIO
import sys
import contextlib
import logging
import asyncio
import aiohttp
import json

from core.agents.agent_base import AgentBase
from core.utils.config_utils import load_config
from core.llm_plugin import LLMPlugin

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class CodeAlchemist(AgentBase):
    """
    The CodeAlchemist is a sophisticated agent designed to handle code generation,
    validation, optimization, and deployment. It leverages LLMs, code analysis tools,
    and potentially even sandboxed environments to produce high-quality, reliable code.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.config = config
        self.llm_plugin = LLMPlugin()
        self.validation_tool_url = self.config.get("validation_tool_url")
        self.optimization_strategies = self.config.get("optimization_strategies", [])
        self.deployment_methods = self.config.get("deployment_methods", [])
        self.knowledge_base = self.load_knowledge_base()
        logging.info(f"CodeAlchemist initialized with config: {config}")

    def load_knowledge_base(self) -> Dict[str, Dict[str, str]]:
        """
        Loads the knowledge base containing code snippets, best practices,
        and other relevant information.
        """
        knowledge_base_path = self.config.get("knowledge_base_path", "data/code_knowledge_base.json")
        try:
            with open(knowledge_base_path, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            logging.warning(f"Knowledge base not found at: {knowledge_base_path}. Using empty knowledge base.")
            return {}
        except json.JSONDecodeError as e:
            logging.error(f"Error decoding knowledge base: {e}. Using empty knowledge base.")
            return {}

    async def execute(self, action: str, **kwargs: Dict[str, Any]) -> Optional[Union[str, Dict[str, Any], bool]]:
        """
        Executes a code-related action.
        """

        if action == "generate_code":
            return await self.generate_code(**kwargs)
        elif action == "validate_code":
            return await self.validate_code(kwargs.get("code"), **kwargs)
        elif action == "optimize_code":
            return await self.optimize_code(kwargs.get("code"), **kwargs)
        elif action == "deploy_code":
            return await self.deploy_code(kwargs.get("code"), kwargs.get("environment"), **kwargs)
        else:
            logging.warning(f"CodeAlchemist: Unknown action: {action}")
            return None

    async def generate_code(self, intent: str, context: Dict[str, Any], constraints: Dict[str, Any]) -> Optional[str]:
        """
        Generates code based on the specified intent, context, and constraints.
        """

        logging.info(f"Generating code for intent: {intent}, context: {context}, constraints: {constraints}")

        # 1. Retrieve Relevant Knowledge
        relevant_knowledge: str = self.get_relevant_knowledge(intent, context)

        # 2. Construct Prompt
        prompt: str = self.construct_generation_prompt(intent, context, constraints, relevant_knowledge)

        # 3. Generate Code using LLM
        generated_code: Optional[str] = await self.generate_code_from_prompt(prompt)

        if generated_code:
            logging.info(f"Generated code: {generated_code}")
            return generated_code
        else:
            logging.error("Code generation failed.")
            return None

    def get_relevant_knowledge(self, intent: str, context: Dict[str, Any]) -> str:
        """
        Retrieves relevant knowledge from the knowledge base.
        """

        logging.info("Retrieving relevant knowledge...")
        relevant_knowledge: List[str] = []

        # Example: Retrieve knowledge based on programming language and task
        language: str = context.get("language", "").lower()
        task_keywords: List[str] = self.extract_keywords(intent)  # Implement extract_keywords
        for keyword in task_keywords:
            if language in self.knowledge_base and keyword in self.knowledge_base[language]:
                relevant_knowledge.append(self.knowledge_base[language][keyword])

        if relevant_knowledge:
            return "\n".join(relevant_knowledge)
        else:
            return "No relevant knowledge found."

    def extract_keywords(self, text: str) -> List[str]:
        """
        Extracts relevant keywords from a text string.
        This is a placeholder for a more sophisticated keyword extraction method.
        """
        # Simple placeholder: Split the text into words
        return re.findall(r'\b\w+\b', text.lower())

    def construct_generation_prompt(self, intent: str, context: Dict[str, Any], constraints: Dict[str, Any], relevant_knowledge: str) -> str:
        """
        Constructs the prompt for code generation.
        """

        prompt: str = f"""
            You are Code Alchemist, a code generation expert.
            I need to generate code that fulfills the following intent: {intent}
            Here is the relevant context: {context}
            The code should adhere to these constraints: {constraints}
            Here is some relevant knowledge you might find useful: {relevant_knowledge}

            Generate the code and explain your reasoning.
            """
        return prompt

    async def generate_code_from_prompt(self, prompt: str) -> Optional[str]:
        """
        Generates code based on the given prompt using an LLM.
        """

        logging.info("Generating code from prompt...")

        llm_result: Optional[str] = await self.llm_plugin.get_completion(prompt)

        if llm_result:
            # Extract code from LLM response (assuming LLM might include explanations)
            code_match: Optional[re.Match[str]] = re.search(r"```(?:\w+\n)?(.*?)```", llm_result, re.DOTALL)
            if code_match:
                generated_code: str = code_match.group(1).strip()
                return generated_code
            else:
                logging.warning("No code block found in LLM response.")
                return None
        else:
            logging.error("LLM failed to provide a result.")
            return None

    async def validate_code(self, code: str, **kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validates code for correctness, efficiency, and security.
        """

        logging.info(f"Validating code: {code}")
        validation_results: Dict[str, Any] = {}

        # 1. Syntax Check
        try:
            ast.parse(code)
            validation_results["syntax_errors"] = None
        except SyntaxError as e:
            validation_results["syntax_errors"] = str(e)

        # 2. Semantic Analysis (Placeholder - LLM or static analysis)
        semantic_analysis_results: Dict[str, Any] = await self.perform_semantic_analysis(code)
        validation_results.update(semantic_analysis_results)

        # 3. Security Analysis (Placeholder - LLM or security tools)
        security_analysis_results: Dict[str, Any] = await self.perform_security_analysis(code)
        validation_results.update(security_analysis_results)

        # 4. Efficiency Analysis (Placeholder - LLM or profiling)
        efficiency_analysis_results: Dict[str, Any] = await self.perform_efficiency_analysis(code)
        validation_results.update(efficiency_analysis_results)

        logging.info(f"Validation results: {validation_results}")
        return validation_results

    async def perform_semantic_analysis(self, code: str) -> Dict[str, Any]:
        """
        Placeholder for semantic analysis (logic errors, etc.) using an LLM.
        """
        prompt: str = f"Analyze the following code for logical errors and correctness:\n```\n{code}\n```\nProvide a JSON output of findings."
        llm_result: Optional[str] = await self.llm_plugin.get_completion(prompt)
        if llm_result:
            try:
                return json.loads(llm_result)
            except json.JSONDecodeError as e:
                logging.error(f"LLM output is not valid JSON for semantic analysis: {e}\nOutput: {llm_result}")
                return {"semantic_errors": f"Invalid JSON from LLM: {e}"}
        else:
            logging.warning("LLM failed to provide semantic analysis.")
            return {"semantic_errors": "LLM failed to provide semantic analysis."}

    async def perform_security_analysis(self, code: str) -> Dict[str, Any]:
        """
        Placeholder for security analysis (vulnerabilities) using an LLM.
        """
        prompt: str = f"Analyze the following code for security vulnerabilities:\n```\n{code}\n```\nProvide a JSON output of findings."
        llm_result: Optional[str] = await self.llm_plugin.get_completion(prompt)
        if llm_result:
            try:
                return json.loads(llm_result)
            except json.JSONDecodeError as e:
                logging.error(f"LLM output is not valid JSON for security analysis: {e}\nOutput: {llm_result}")
                return {"security_vulnerabilities": f"Invalid JSON from LLM: {e}"}
        else:
            logging.warning("LLM failed to provide security analysis.")
            return {"security_vulnerabilities": "LLM failed to provide security analysis."}

    async def perform_efficiency_analysis(self, code: str) -> Dict[str, Any]:
        """
        Placeholder for efficiency analysis (performance) using an LLM.
        """
        prompt: str = f"Analyze the following code for efficiency and performance:\n```\n{code}\n```\nProvide a JSON output of findings and suggestions."
        llm_result: Optional[str] = await self.llm_plugin.get_completion(prompt)
        if llm_result:
            try:
                return json.loads(llm_result)
            except json.JSONDecodeError as e:
                logging.error(f"LLM output is not valid JSON for efficiency analysis: {e}\nOutput: {llm_result}")
                return {"efficiency_issues": f"Invalid JSON from LLM: {e}"}
        else:
            logging.warning("LLM failed to provide efficiency analysis.")
            return {"efficiency_issues": "LLM failed to provide efficiency analysis."}

    async def optimize_code(self, code: str, optimization_strategies: Optional[List[str]] = None) -> Optional[str]:
        """
        Optimizes code for performance and maintainability.
        """

        logging.info(f"Optimizing code: {code} with strategies: {optimization_strategies}")

        if not optimization_strategies:
            optimization_strategies = self.optimization_strategies  # Use default strategies

        optimized_code: str = code
        for strategy in optimization_strategies:
            optimized_code = await self.apply_optimization_strategy(optimized_code, strategy)

        if optimized_code != code:
            logging.info(f"Optimized code: {optimized_code}")
            return optimized_code
        else:
            logging.info("No optimization strategies applied.")
            return code

    async def apply_optimization_strategy(self, code: str, strategy: str) -> str:
        """
        Applies a specific optimization strategy to the code.
        This is a placeholder for more advanced optimization techniques.
        """

        prompt: str = f"Apply the '{strategy}' optimization strategy to the following code:\n```\n{code}\n```\nProvide the optimized code."
        llm_result: Optional[str] = await self.llm_plugin.get_completion(prompt)

        if llm_result:
            # Extract code from LLM response (assuming LLM might include explanations)
            code_match: Optional[re.Match[str]] = re.search(r"```(?:\w+\n)?(.*?)```", llm_result, re.DOTALL)
            if code_match:
                optimized_code: str = code_match.group(1).strip()
                return optimized_code
            else:
                logging.warning(f"No code block found after applying optimization strategy: {strategy}")
                return code  # Return original code if no code block
        else:
            logging.warning(f"LLM failed to apply optimization strategy: {strategy}. Returning original code.")
            return code  # Return original code if LLM fails

    async def deploy_code(self, code: str, deployment_method: str, **kwargs: Dict[str, Any]) -> bool:
        """
        Assists in deploying code using various methods.
        """

        logging.info(f"Deploying code using method: {deployment_method}")

        if deployment_method not in self.deployment_methods:
            logging.error(f"Unsupported deployment method: {deployment_method}")
            return False

        if deployment_method == "local_file":
            return self.deploy_to_local_file(code, **kwargs)
        elif deployment_method == "api_endpoint":
            return await self.deploy_to_api_endpoint(code, **kwargs)
        elif deployment_method == "cloud_function":
            return await self.deploy_to_cloud_function(code, **kwargs)
        else:
            logging.error(f"CodeAlchemist: Unknown deployment method: {deployment_method}")
            return False

    def deploy_to_local_file(self, code: str, file_path: str) -> bool:
        """
        Deploys code by saving it to a local file.
        """

        try:
            with open(file_path, "w") as f:
                f.write(code)
            logging.info(f"Code deployed to local file: {file_path}")
            return True
        except Exception as e:
            logging.error(f"Error deploying code to local file '{file_path}': {e}")
            return False

    async def deploy_to_api_endpoint(self, code: str, api_endpoint: str) -> bool:
        """
        Deploys code by sending it to an API endpoint.
        """

        headers: Dict[str, str] = {"Content-Type": "application/json"}
        payload: Dict[str, str] = {"code": code}

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(api_endpoint, headers=headers, data=json.dumps(payload)) as response:
                    if response.status == 200:
                        logging.info(f"Code deployed to API endpoint: {api_endpoint}")
                        return True
                    else:
                        error_message: str = await response.text()
                        logging.error(f"Error deploying code to API endpoint '{api_endpoint}': {error_message}")
                        return False
        except aiohttp.ClientError as e:
            logging.error(f"Error connecting to API endpoint '{api_endpoint}': {e}")
            return False

    async def deploy_to_cloud_function(self, code: str, cloud_function_name: str) -> bool:
        """
        Deploys code to a cloud function (e.g., AWS Lambda, Google Cloud Function).
        This is a placeholder and would require specific cloud provider SDKs.
        """

        logging.warning(f"Cloud function deployment is not yet implemented: {cloud_function_name}")
        return False  # Placeholder


    async def run(self):
        """
        Main execution loop for the CodeAlchemist agent.
        This would typically involve fetching requests from a queue or API.
        For demonstration, it runs a single example.
        """

        # Example usage:
        intent: str = "Generate a Python function to calculate the nth Fibonacci number efficiently."
        context: Dict[str, str] = {"language": "python", "environment": "local"}
        constraints: Dict[str, str] = {"performance": "optimized", "memory_usage": "low", "security": "safe"}

        generated_code: Optional[str] = await self.generate_code(intent, context, constraints)
        if generated_code:
            validation_results: Dict[str, Any] = await self.validate_code(generated_code)
            if not validation_results.get("syntax_errors") and not validation_results.get("semantic_errors") and not validation_results.get("security_vulnerabilities"):
                optimized_code: Optional[str] = await self.optimize_code(generated_code, ["performance", "readability"])
                if optimized_code:
                    success: bool = await self.deploy_code(optimized_code, "local_file", file_path="fibonacci.py")
                    if success:
                        logging.info("Code generation, validation, optimization, and deployment successful!")
                    else:
                        logging.error("Code deployment failed.")
                else:
                    logging.error("Code optimization failed.")
            else:
                logging.error(f"Code validation failed: {validation_results}")
        else:
            logging.error("Code generation failed.")


# Example usage
if __name__ == "__main__":
    config: Dict[str, Any] = {
        "knowledge_base_path": "data/code_knowledge_base.json",
        "validation_tool_url": "http://localhost:8000/validate",
        "optimization_strategies": ["performance", "readability"],
        "deployment_methods": ["local_file", "api_endpoint"],
    }  # Load configuration
    code_alchemist: CodeAlchemist = CodeAlchemist(config)
    asyncio.run(code_alchemist.run())  # Use asyncio.run for the async run method


