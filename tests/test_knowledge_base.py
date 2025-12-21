# tests/test_knowledge_base.py

import unittest

from core.system.knowledge_base import KnowledgeBase


class TestKnowledgeBase(unittest.TestCase):

    def setUp(self):
        self.kb = KnowledgeBase()

    def test_query_existing_key(self):
        self.kb.update("test_key", "test_value")
        result = self.kb.query("test_key")
        self.assertEqual(result, "test_value")

    def test_query_nonexistent_key(self):
        result = self.kb.query("nonexistent_key")
        self.assertIsNone(result)

    def test_update(self):
        self.kb.update("test_key", "initial_value")
        self.kb.update("test_key", "updated_value")
        result = self.kb.query("test_key")
        self.assertEqual(result, "updated_value")

    def test_query_case_insensitive(self):
        self.kb.update("TeSt_KeY", "test_value")
        result = self.kb.query("test_key")
        self.assertEqual(result, "test_value")

if __name__ == '__main__':
    unittest.main()
