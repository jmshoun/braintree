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
