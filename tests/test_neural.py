import unittest

from context import neural


class NeuralTest(unittest.TestCase):
    def test_model_construction(self):
        model_ = neural.NeuralModel(30)
        self.assertIsNotNone(model_)
