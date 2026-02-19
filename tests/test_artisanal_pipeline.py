import unittest
import sys
import os
import re

# Add repo root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.ingestion.semantic_chunker import SemanticChunker
from core.evaluation.llm_judge import ConvictionScorer

# Helper function from the script to test independently
def clean_html(text):
    if not text:
        return ""
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text).strip()

class TestArtisanalPipeline(unittest.TestCase):
    def test_chunker(self):
        chunker = SemanticChunker()
        text = "Hello.\n\nWorld."
        chunks = chunker.split_text(text)
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0], "Hello.")
        self.assertEqual(chunks[1], "World.")

    def test_chunker_empty(self):
        chunker = SemanticChunker()
        self.assertEqual(chunker.split_text(""), [])

    def test_judge_evidence(self):
        judge = ConvictionScorer()
        text = "Growth is 5%."
        score = judge.score(text)
        self.assertEqual(score['score'], 0.7)

    def test_judge_hedging(self):
        judge = ConvictionScorer()
        text = "Maybe it will rain."
        score = judge.score(text)
        self.assertEqual(score['score'], 0.3)

    def test_clean_html(self):
        html = "<p>Hello <b>World</b></p>"
        cleaned = clean_html(html)
        self.assertEqual(cleaned, "Hello World")

if __name__ == '__main__':
    unittest.main()
