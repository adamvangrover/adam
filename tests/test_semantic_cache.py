import unittest
import shutil
import os
import time
from core.infrastructure.semantic_cache import SemanticCache

class TestSemanticCache(unittest.TestCase):

    def setUp(self):
        self.test_dir = "tests/temp_cache"
        self.cache = SemanticCache(cache_dir=self.test_dir)

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_cache_miss_hit(self):
        prompt = "Analyze risk"
        data = {"revenue": 100}
        data_hash = SemanticCache.compute_data_hash(data)
        model = "gpt-4"

        # 1. Miss
        result = self.cache.get(prompt, data_hash, model)
        self.assertIsNone(result)

        # 2. Set
        output = {"risk": "high"}
        self.cache.set(prompt, data_hash, model, output)

        # 3. Hit
        result = self.cache.get(prompt, data_hash, model)
        self.assertEqual(result, output)

    def test_context_sensitivity(self):
        prompt = "Analyze risk"
        model = "gpt-4"

        data1 = {"revenue": 100}
        hash1 = SemanticCache.compute_data_hash(data1)

        data2 = {"revenue": 200}
        hash2 = SemanticCache.compute_data_hash(data2)

        self.cache.set(prompt, hash1, model, {"risk": "low"})

        # Should be None for data2
        self.assertIsNone(self.cache.get(prompt, hash2, model))

if __name__ == '__main__':
    unittest.main()
