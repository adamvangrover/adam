
import unittest
import sys
import os

# Ensure importability
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.data_processing.utils import GoldStandardScrubber, FileHandlers

class TestCoreDataProcessingUtils(unittest.TestCase):

    def test_clean_text(self):
        text = "This is a text.\n\n\n\nIt has many newlines."
        cleaned = GoldStandardScrubber.clean_text(text)
        expected = "This is a text.\n\nIt has many newlines."
        self.assertEqual(cleaned, expected)

    def test_extract_metadata(self):
        content = "This is a text with #hashtag and #another. It also mentions Apple and Microsoft."
        metadata = GoldStandardScrubber.extract_metadata(content)

        self.assertIn("tags", metadata)
        tags = metadata["tags"]
        self.assertIn("#hashtag", tags)
        self.assertIn("#another", tags)

        self.assertIn("potential_entities", metadata)
        entities = metadata["potential_entities"]
        self.assertIn("Apple", entities)
        self.assertIn("Microsoft", entities)

    def test_handle_markdown_title(self):
        markdown_text = """
# My Title
This is some content.
"""
        artifact = FileHandlers.handle_markdown("dummy.md", markdown_text)
        self.assertEqual(artifact.title, "My Title")

    def test_handle_markdown_no_title(self):
        markdown_text = "Just some text without a header."
        artifact = FileHandlers.handle_markdown("dummy.md", markdown_text)
        self.assertEqual(artifact.title, "dummy.md")

if __name__ == '__main__':
    unittest.main()
