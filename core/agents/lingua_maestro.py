# core/agents/lingua_maestro.py

import asyncio
import logging
from typing import Any, Dict, Optional

from core.agents.agent_base import AgentBase

try:
    from langchain.utilities import GoogleSearchAPIWrapper
except ImportError:
    try:
        from langchain_community.utilities import GoogleSearchAPIWrapper
    except ImportError:
        GoogleSearchAPIWrapper = None

try:
    from transformers import pipeline
except ImportError:
    pipeline = None

try:
    from nltk.sentiment import SentimentIntensityAnalyzer
except ImportError:
    SentimentIntensityAnalyzer = None

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LinguaMaestro(AgentBase):
    """
    Agent specializing in Natural Language Processing, Translation, and Communication Adaptation.
    """

    def __init__(self, config: Dict[str, Any], kernel: Optional[Any] = None):
        """
        Initializes LinguaMaestro.

        Args:
            config (dict): Configuration dictionary.
            kernel (Optional[Any]): Semantic Kernel instance.
        """
        super().__init__(config, kernel=kernel)

        try:
            self.search_api = GoogleSearchAPIWrapper() if GoogleSearchAPIWrapper else None
        except Exception as e:
            logger.warning(f"Failed to init GoogleSearchAPIWrapper: {e}")
            self.search_api = None

        try:
            if pipeline:
                # Use a small model to avoid huge downloads if possible, or catch error
                self.translator = pipeline("translation_en_to_fr", model="t5-small")
            else:
                self.translator = None
        except Exception as e:
            logger.warning(f"Failed to init translation pipeline: {e}")
            self.translator = None

        try:
            self.sentiment_analyzer = SentimentIntensityAnalyzer() if SentimentIntensityAnalyzer else None
        except Exception as e:
            logger.warning(f"Failed to init SentimentIntensityAnalyzer: {e}")
            self.sentiment_analyzer = None

    async def execute(self, *args, **kwargs):
        """
        Executes NLP tasks.

        Tasks:
        - "translate": Translates text.
        - "analyze_tone": Analyzes text tone.
        - "adapt_style": Adapts communication style.
        """
        task = kwargs.get('task')
        text = kwargs.get('text', '')

        logger.info(f"LinguaMaestro executing task: {task}")

        if task == "translate":
            target_lang = kwargs.get('target_language', 'fr')
            return self.translate_text(text, target_lang)

        elif task == "analyze_tone":
            return self.analyze_tone(text)

        elif task == "adapt_style":
            recipient = kwargs.get('recipient')
            return self.adapt_communication(text, recipient)

        else:
            return {"error": f"Unknown task: {task}"}

    def detect_language(self, text, **kwargs):
        return "en"

    def translate_text(self, text, target_language, **kwargs):
        if self.translator:
            try:
                return self.translator(text)[0]['translation_text']
            except Exception as e:
                logger.error(f"Translation failed: {e}")
        return f"[MOCK TRANSLATION to {target_language}]: {text}"

    def adapt_communication(self, message, recipient, **kwargs):
        return message

    def translate_code(self, code, target_language, **kwargs):
        return code

    def analyze_tone(self, text, **kwargs):
        if self.sentiment_analyzer:
            try:
                scores = self.sentiment_analyzer.polarity_scores(text)
                if scores['compound'] >= 0.05:
                    return 'positive'
                if scores['compound'] <= -0.05:
                    return 'negative'
                return 'neutral'
            except:
                pass
        return 'neutral'

    def recognize_persona(self, text, **kwargs):
        return 'unknown'

    def learn_style_and_preferences(self, interactions, **kwargs):
        pass

    def adapt_behavior(self, tone, persona, preferences, **kwargs):
        pass

if __name__ == "__main__":
    agent = LinguaMaestro({})
    async def main():
        res = await agent.execute(task="translate", text="Hello world")
        print(res)
    asyncio.run(main())
