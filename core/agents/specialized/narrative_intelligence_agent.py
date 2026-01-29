from typing import Any, Dict, List, Tuple
import logging
from collections import Counter
import re
from core.agents.market_sentiment_agent import MarketSentimentAgent

class NarrativeIntelligenceAgent(MarketSentimentAgent):
    """
    Protocol: ADAM-V-NEXT
    Specialized agent that moves beyond simple sentiment scoring to identify
    emerging thematic narratives (e.g., 'AI Bubble', 'Energy Crisis').
    """

    def __init__(self, config: Dict[str, Any], **kwargs):
        super().__init__(config, **kwargs)
        self.narrative_threshold = config.get('narrative_threshold', 3) # Min occurrences to be a narrative

    async def execute(self, **kwargs) -> Dict[str, Any]:
        """
        Executes narrative analysis, extending the base sentiment analysis.
        """
        logging.info("NarrativeIntelligenceAgent execution started.")

        # 1. Run Base Logic (Additive)
        base_result = await super().execute(**kwargs)

        # 2. Add Narrative Layer
        narratives = await self.identify_narratives()

        # 3. Enrich Result
        base_result['narratives'] = narratives
        base_result['dominant_narrative'] = narratives[0] if narratives else "Market noise - No clear signal"

        return base_result

    async def identify_narratives(self) -> List[Dict[str, Any]]:
        """
        Identifies emerging narratives by analyzing keyword clusters across sources.
        """
        # In a real scenario, we would pull raw text from the APIs.
        # Since the base class APIs (Simulated...) might return structured data,
        # we will simulate the text extraction step for this additive demo.

        # Simulated "Mind Stream" from sources
        raw_stream = [
            "AI stocks are surging due to new chip demand",
            "Tech sector rally continues",
            "Energy prices stabilize despite geopolitical tension",
            "AI regulation fears might cap growth",
            "Inflation data comes in hot",
            "Fed signals rate pause",
            "AI models are consuming massive power"
        ]

        # Simple Frequency Analysis (Robust, no LLM dependency)
        # Tokenize and filter stops
        stop_words = {'are', 'due', 'to', 'new', 'despite', 'might', 'comes', 'in', 'is'}
        tokens = []
        for text in raw_stream:
            words = re.findall(r'\w+', text.lower())
            tokens.extend([w for w in words if w not in stop_words and len(w) > 3])

        freq = Counter(tokens)

        # Cluster into Narratives
        # Any token with frequency >= threshold is a "Core Theme"
        themes = [word for word, count in freq.most_common(5) if count >= 2]

        narratives = []
        for theme in themes:
            # Generate a "Headline" for the theme
            related_texts = [t for t in raw_stream if theme in t.lower()]
            sentiment_score = 0.5 # Default neutral

            # Simple sentiment proxy for the theme
            pos_words = {'surging', 'rally', 'growth', 'hot'}
            neg_words = {'fears', 'tension', 'cap'}

            score = 0
            for text in related_texts:
                if any(w in text.lower() for w in pos_words): score += 1
                if any(w in text.lower() for w in neg_words): score -= 1

            if score > 0: sentiment = "BULLISH"
            elif score < 0: sentiment = "BEARISH"
            else: sentiment = "NEUTRAL"

            narratives.append({
                "theme": theme.upper(),
                "headline": f"{theme.title()} Narrative Emerging",
                "volume": freq[theme],
                "sentiment": sentiment,
                "sources": len(related_texts)
            })

        if not narratives:
            # Fallback
            narratives.append({
                "theme": "UNCERTAINTY",
                "headline": "Mixed Signals Dominating",
                "volume": 0,
                "sentiment": "NEUTRAL",
                "sources": 0
            })

        return narratives
