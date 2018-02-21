import unittest

import numpy as np

from context import braintree
from context import data
from context import concrete

concrete_data = data.BrainTreeData(concrete[:, :7], concrete[:, 7:])
concrete_data.shuffle()
concrete_train, concrete_test = concrete_data.split(0.7)


class BrainTreeTest(unittest.TestCase):
    def test_model_construction(self):
        model = braintree.BrainTree(num_trees=25)
        self.assertIsNotNone(model)

    def test_model_scoring(self):
        model = braintree.BrainTree(num_trees=15)
        with self.assertRaises(braintree.NotFitError):
            model.predict(concrete_test)
        model.fit(concrete_train, concrete_test)
        predictions = model.predict(concrete_test)
        self.assertEqual(len(predictions), 31)
