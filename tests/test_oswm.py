import unittest
from core.oswm.model import OSWMTransformer

class TestOSWM(unittest.TestCase):
    def test_forward_shape(self):
        model = OSWMTransformer(vocab_size=100, d_model=32)
        input_ids = [1, 2, 3]
        logits = model.forward(input_ids)

        self.assertEqual(len(logits), 100)
        self.assertIsInstance(logits[0], float)

if __name__ == '__main__':
    unittest.main()
