import unittest

from context import neural
from context import data
from context import concrete

concrete_data = data.BrainTreeData(concrete[:, :7], concrete[:, 7:])
concrete_data.shuffle()
concrete_train, concrete_test = concrete_data.split(0.7)


class NeuralTest(unittest.TestCase):
    def test_model_construction(self):
        model_ = neural.NeuralModel(30)
        self.assertIsNotNone(model_)

    def test_model_predictions(self):
        model_ = neural.NeuralModel(7)
        predictions = model_.predict(concrete_train)
        self.assertListEqual(predictions.tolist(), [0.] * 72)
