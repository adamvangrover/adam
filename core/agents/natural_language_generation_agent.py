# core/agents/natural_language_generation_agent.py

import logging
import asyncio
from typing import Dict, Any, Optional
from transformers import pipeline
from core.agents.agent_base import AgentBase

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NaturalLanguageGenerationAgent(AgentBase):
    """
    Agent responsible for generating natural language text.
    """

    def __init__(self, config: Dict[str, Any], kernel: Optional[Any] = None):
        """
        Initializes the NLG Agent.

        Args:
            config (dict): Configuration dictionary.
            kernel (Optional[Any]): Semantic Kernel instance.
        """
        super().__init__(config, kernel=kernel)
        self.model_name = self.config.get('model_name', 'gpt2')
        try:
            self.generator = pipeline('text-generation', model=self.model_name)
        except Exception as e:
            logger.error(f"Failed to load generation model: {e}")
            self.generator = None

    async def execute(self, *args, **kwargs):
        """
        Executes NLG tasks.

        Tasks:
        - "text": Generates text from prompt.
        - "summary": Summarizes data.
        - "report": Generates a report.
        """
        output_type = kwargs.get('output_type')
        data = kwargs.get('data')

        logger.info(f"NaturalLanguageGenerationAgent executing output_type: {output_type}")

        try:
            # Run blocking model calls in executor
            loop = asyncio.get_running_loop()

            if output_type == "text":
                return await loop.run_in_executor(None, self.generate_text, data, kwargs)
            elif output_type == "summary":
                return await loop.run_in_executor(None, self.summarize_data, data, kwargs)
            elif output_type == "report":
                return await loop.run_in_executor(None, self.generate_report, data, kwargs.get('report_type'), kwargs)
            else:
                return {"error": "Invalid output type."}
        except Exception as e:
            logger.error(f"NLG error: {e}")
            return {"error": str(e)}

    def generate_text(self, prompt, kwargs):
        """
        Generates text based on the given prompt and parameters.
        """
        if not self.generator:
            return "Model not initialized."
        try:
            # Clean kwargs for pipeline
            gen_kwargs = {k: v for k, v in kwargs.items() if k in ['max_length', 'num_return_sequences']}
            generated_text = self.generator(prompt, **gen_kwargs)
            return generated_text
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return str(e)

    def summarize_data(self, data, kwargs):
        """
        Summarizes the given data into a concise and informative text.
        """
        prompt = f"Summarize the following data:\n\n{data}\n\nSummary:"
        return self.generate_text(prompt, kwargs)

    def generate_report(self, data, report_type, kwargs):
        """
        Generates a report of the specified type based on the given data.
        """
        if report_type == "market_sentiment":
            prompt = f"Generate market sentiment report for: {data}"
            return self.generate_text(prompt, kwargs)
        elif report_type == "financial_analysis":
            prompt = f"Generate financial analysis report for: {data}"
            return self.generate_text(prompt, kwargs)

        return "Unknown report type."

if __name__ == "__main__":
    agent = NaturalLanguageGenerationAgent({'model_name': 'gpt2'}) # Use 'distilgpt2' for faster testing if needed
    async def main():
        # Example data
        data = "Apple revenue 100B, Net Income 20B."
        res = await agent.execute(output_type="summary", data=data, max_length=50)
        print(res)
    asyncio.run(main())
