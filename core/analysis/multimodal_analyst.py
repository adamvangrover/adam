from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
from core.analysis.base_analyzer import BaseFinancialAnalyzer
from core.llm_plugin import LLMPlugin
import logging
import asyncio
import os

logger = logging.getLogger(__name__)

class AudioAnalysisResult(BaseModel):
    summary: str = Field(..., description="Summary of the audio content.")
    sentiment: str = Field(..., description="Overall sentiment (Positive, Negative, Neutral).")
    key_quotes: list[str] = Field(..., description="Key quotes extracted from the audio.")
    speaker_intent: str = Field(..., description="Inferred intent of the speaker.")

class AudioFinancialAnalyzer(BaseFinancialAnalyzer):
    """
    Analyzer for financial audio (Earnings Calls, Interviews) using Gemini 1.5 Pro.
    """

    def __init__(self, llm_plugin: Optional[LLMPlugin] = None):
        self.llm = llm_plugin or LLMPlugin()

    async def analyze_report(self, report_text: str, context: str = "") -> BaseModel:
        # Not applicable for audio analyzer, but implementing base abstract method
        raise NotImplementedError("Use analyze_audio instead.")

    async def analyze_image(self, image_path: str, context: str = "") -> Dict[str, Any]:
        # Not applicable
        raise NotImplementedError("Use analyze_audio instead.")

    async def analyze_audio(self, audio_path: str, context: str = "") -> AudioAnalysisResult:
        """
        Analyzes an audio file.
        """
        if not os.path.exists(audio_path) and "mock" not in audio_path.lower():
             logger.warning(f"Audio file {audio_path} not found. Proceeding if mock.")

        prompt = (
            f"You are a financial analyst listening to an earnings call or interview.\n"
            f"Context: {context}\n"
            f"Analyze the audio and provide:\n"
            f"1. A concise summary.\n"
            f"2. The overall sentiment.\n"
            f"3. Key quotes that indicate future performance.\n"
            f"4. The speaker's underlying intent (confidence, hesitation, deflection).\n\n"
            f"Return the result as a JSON object."
        )

        try:
            loop = asyncio.get_running_loop()
            response_text = await loop.run_in_executor(
                None,
                lambda: self.llm.generate_audio_analysis(prompt, audio_path)
            )

            # Simple parsing for now - in production use structured generation if supported for audio
            # Here we assume the LLM returns a JSON string or we mock it in the plugin
            if "Mock" in response_text and "{" not in response_text:
                 # Manually construct mock obj if the mock return was simple text
                 return AudioAnalysisResult(
                     summary=response_text,
                     sentiment="Positive",
                     key_quotes=["We are confident."],
                     speaker_intent="Confidence"
                 )

            import json
            clean_text = response_text.replace("```json", "").replace("```", "").strip()
            data = json.loads(clean_text)
            return AudioAnalysisResult(**data)

        except Exception as e:
            logger.error(f"Audio analysis failed: {e}")
            return AudioAnalysisResult(
                summary=f"Error: {e}",
                sentiment="Unknown",
                key_quotes=[],
                speaker_intent="Error"
            )

class VideoFinancialAnalyzer(BaseFinancialAnalyzer):
    """
    Analyzer for video content. Currently a wrapper around AudioFinancialAnalyzer
    plus image analysis of key frames (future).
    """
    def __init__(self, llm_plugin: Optional[LLMPlugin] = None):
        self.audio_analyzer = AudioFinancialAnalyzer(llm_plugin)
        self.llm = llm_plugin or LLMPlugin()

    async def analyze_report(self, report_text: str, context: str = "") -> BaseModel:
        raise NotImplementedError

    async def analyze_image(self, image_path: str, context: str = "") -> Dict[str, Any]:
        raise NotImplementedError

    async def analyze_video(self, video_path: str, context: str = "") -> Dict[str, Any]:
        """
        Stub for video analysis.
        """
        # 1. Extract audio (hypothetically) -> Analyze Audio
        audio_res = await self.audio_analyzer.analyze_audio(video_path, context=context)

        # 2. Extract frames -> Analyze Images (not impl yet)

        return {
            "audio_analysis": audio_res.model_dump(),
            "visual_analysis": "Not implemented (requires frame extraction)"
        }
