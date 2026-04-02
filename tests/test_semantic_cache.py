import unittest
import shutil
import tempfile
import time
from core.infrastructure.semantic_cache import SemanticCache

class TestSemanticCache(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.cache = SemanticCache(cache_dir=self.test_dir, memory_capacity=2)

    def tearDown(self):
        shutil.rmtree(self.test_dir)
        self.cache.clear()

    def test_compute_data_hash(self):
        data = {"key": "value"}
        data_hash = SemanticCache.compute_data_hash(data)
        self.assertIsInstance(data_hash, str)
        self.assertEqual(len(data_hash), 64)  # SHA-256 hash length

    def test_set_and_get(self):
        prompt = "test prompt"
        context_hash = "test_hash"
        model_id = "test_model"
        output = {"result": "success"}

        self.cache.set(prompt, context_hash, model_id, output)
        cached_output = self.cache.get(prompt, context_hash, model_id)

        self.assertEqual(cached_output, output)

    def test_cache_miss(self):
        cached_output = self.cache.get("unknown", "hash", "model")
        self.assertIsNone(cached_output)

    def test_deterministic_hash(self):
        data1 = {"a": 1, "b": 2}
        data2 = {"b": 2, "a": 1}

        hash1 = SemanticCache.compute_data_hash(data1)
        hash2 = SemanticCache.compute_data_hash(data2)

        self.assertEqual(hash1, hash2)

    def test_memory_capacity(self):
        # We set capacity to 2 in setUp
        self.cache.set("p1", "c1", "m1", {"r": 1})
        self.cache.set("p2", "c2", "m2", {"r": 2})
        self.cache.set("p3", "c3", "m3", {"r": 3})  # This should evict p1 from memory

        # Memory should only have 2 items
        self.assertEqual(len(self.cache._memory_cache), 2)

        # p1 should still be accessible from disk fallback
        self.assertEqual(self.cache.get("p1", "c1", "m1"), {"r": 1})

        # Getting p1 should bring it back into memory, evicting p2
        self.assertEqual(len(self.cache._memory_cache), 2)

    def test_clear_cache(self):
        self.cache.set("test", "hash", "model", {"res": "val"})
        self.assertEqual(len(self.cache._memory_cache), 1)
        self.cache.clear()
        self.assertEqual(len(self.cache._memory_cache), 0)

if __name__ == '__main__':
    unittest.main()
