# core/agents/prompt_tuner.py

try:
    from langchain.prompts import PromptTemplate
except ImportError:
    try:
        from langchain_core.prompts import PromptTemplate
    except ImportError:
        PromptTemplate = None

from core.utils.data_utils import send_message
from typing import List, Dict, Any, Optional
import re
import logging
import asyncio
import json
from core.agents.agent_base import AgentBase

try:
    from textblob import TextBlob
except ImportError:
    TextBlob = None

try:
    import spacy
except ImportError:
    spacy = None

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PromptTuner(AgentBase):
    """
    Refines and optimizes prompts for communication and analysis.
    """

    def __init__(self, config: Dict[str, Any], kernel: Optional[Any] = None):
        """
        Initializes the Prompt Tuner.

        Args:
            config (dict): Configuration dictionary.
            kernel (Optional[Any]): Semantic Kernel instance.
        """
        super().__init__(config, kernel=kernel)
        self.user_feedback_enabled = self.config.get('user_feedback_enabled', False)
        self.hallucination_detection_enabled = self.config.get('hallucination_detection_enabled', False)
        self.knowledge_base = self.config.get('knowledge_base', {})
        self.nlp = None

        if spacy:
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except Exception as e:
                logger.warning(f"Failed to load spacy model 'en_core_web_sm': {e}. NLP features disabled.")
        else:
            logger.warning("Spacy not installed. NLP features disabled.")

    async def execute(self, *args, **kwargs):
        """
        Executes prompt tuning tasks.

        Tasks:
        - "analyze": Analyzes prompt quality.
        - "enhance": Enhances prompt for machine readability.
        - "contextualize": Injects context.
        """
        task = kwargs.get('task')
        prompt = kwargs.get('prompt')

        logger.info(f"PromptTuner executing task: {task}")

        if task == "analyze":
            return self.analyze_prompt(prompt)

        elif task == "enhance":
            return self.enhance_machine_readability(prompt)

        elif task == "contextualize":
            context = kwargs.get('context', {})
            return self.contextualize_prompt(prompt, context)

        else:
            return {"error": f"Unknown task: {task}"}

    def analyze_prompt(self, prompt: str, **kwargs) -> Dict:
        """
        Analyzes a prompt to identify potential issues or areas for improvement.
        """
        analysis_results = {
            'clarity': self._analyze_clarity(prompt),
            'conciseness': self._analyze_conciseness(prompt),
            'relevance': self._analyze_relevance(prompt),
            'sentiment': self._analyze_sentiment(prompt),
            'keywords': self._extract_keywords(prompt),
            'entities': self._extract_entities(prompt),
        }
        return analysis_results

    def _analyze_clarity(self, prompt: str) -> str:
        if not self.nlp:
            return 'medium'
        doc = self.nlp(prompt)
        passive_count = sum(1 for token in doc if token.dep_ == 'nsubjpass')
        if passive_count > 1:
            return 'medium'
        else:
            return 'high'

    def _analyze_conciseness(self, prompt: str) -> str:
        if len(prompt.split()) > 20:
            return 'low'
        elif len(prompt.split()) > 10:
            return 'medium'
        else:
            return 'high'

    def _analyze_relevance(self, prompt: str) -> str:
        if 'financial analysis' in prompt.lower() or 'investment' in prompt.lower():
            return 'high'
        else:
            return 'medium'

    def _analyze_sentiment(self, prompt: str) -> str:
        if not TextBlob:
            return 'neutral'
        blob = TextBlob(prompt)
        if blob.sentiment.polarity > 0.1:
            return 'positive'
        elif blob.sentiment.polarity < -0.1:
            return 'negative'
        else:
            return 'neutral'

    def _extract_keywords(self, prompt: str) -> List[str]:
        if not self.nlp:
            return list(set(prompt.split()))
        doc = self.nlp(prompt)
        keywords = [token.text for token in doc if token.pos_ in ('NOUN', 'ADJ', 'VERB')]
        return keywords

    def _extract_entities(self, prompt: str) -> List[str]:
        if not self.nlp:
            return []
        doc = self.nlp(prompt)
        entities = [ent.text for ent in doc.ents]
        return entities

    def contextualize_prompt(self, prompt: str, context: Dict, **kwargs) -> str:
        """
        Incorporates relevant context and information into a prompt.
        """
        contextualized_prompt = prompt
        for key, value in context.items():
            contextualized_prompt = contextualized_prompt.replace(f"{{{key}}}", str(value))
        return contextualized_prompt

    def enhance_machine_readability(self, prompt: str, **kwargs) -> str:
        """
        Enhances a prompt for better machine readability.
        """
        enhanced_prompt = re.sub(r'[^\w\s]', '', prompt).lower()
        return enhanced_prompt

if __name__ == "__main__":
    agent = PromptTuner({})
    async def main():
        prompt = "Analyze the performance of Tesla."
        res = await agent.execute(task="analyze", prompt=prompt)
        print(res)
    asyncio.run(main())
