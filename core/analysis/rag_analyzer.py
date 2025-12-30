from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from core.analysis.base_analyzer import BaseFinancialAnalyzer
from core.rag.rag_engine import RAGEngine
from core.llm_plugin import LLMPlugin
import logging
import asyncio

logger = logging.getLogger(__name__)

class RAGAnalysisResult(BaseModel):
    summary: str = Field(..., description="Summary of the analyzed documents.")
    key_findings: List[str] = Field(..., description="Key findings extracted via RAG.")
    answers: Dict[str, str] = Field(..., description="Answers to specific strategic questions.")

class RAGFinancialAnalyzer(BaseFinancialAnalyzer):
    """
    Analyzer that uses Retrieval Augmented Generation to process
    large or multiple financial documents.
    """

    def __init__(self, llm_plugin: Optional[LLMPlugin] = None):
        self.llm = llm_plugin or LLMPlugin()
        self.rag_engine = RAGEngine(llm_plugin=self.llm)

    async def analyze_report(self, report_text: str, context: str = "") -> RAGAnalysisResult:
        """
        Analyzes a report by ingesting it into RAG and asking strategic questions.
        """
        # 1. Ingest document
        self.rag_engine.add_document(report_text, metadata={"context": context})

        # 2. Define strategic questions
        questions = [
            "What are the primary risks mentioned?",
            "What is the management's outlook for the next fiscal year?",
            "Are there any regulatory concerns?",
            "What are the key drivers of revenue growth?"
        ]

        # 3. Query RAG engine for each question
        answers = {}
        for q in questions:
            # RAG engine query is sync, run in thread if needed, but it's fast enough for now or we can make it async later
            # For strict async adherence:
            loop = asyncio.get_running_loop()
            ans = await loop.run_in_executor(None, lambda: self.rag_engine.query(q))
            answers[q] = ans

        # 4. Generate synthesis
        findings_str = "\n".join([f"Q: {q}\nA: {a}" for q, a in answers.items()])

        synthesis_prompt = (
            f"Based on the following findings from a financial report, provide a cohesive summary and list key takeaways.\n\n"
            f"Findings:\n{findings_str}\n\n"
            f"Context: {context}\n"
        )

        loop = asyncio.get_running_loop()
        summary_text = await loop.run_in_executor(
            None,
            lambda: self.llm.generate_text(synthesis_prompt, task="summarization")
        )

        return RAGAnalysisResult(
            summary=summary_text,
            key_findings=[a for a in answers.values()],
            answers=answers
        )

    async def analyze_image(self, image_path: str, context: str = "") -> Dict[str, Any]:
        """
        Multimodal RAG is advanced. For now, delegate to basic image analysis
        or use the image description as a document.
        """
        # Get image description from LLM Vision
        description = self.llm.generate_multimodal(
            prompt="Describe this financial chart or table in detail for indexing.",
            image_path=image_path
        )

        # Add description to RAG
        self.rag_engine.add_document(description, metadata={"source_image": image_path, "context": context})

        return {
            "status": "indexed",
            "description": description
        }
