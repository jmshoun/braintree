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
        model.fit(concrete_train, concrete_test, print_every=250)
        predictions = model.predict(concrete_test)
        self.assertEqual(len(predictions), 31)

    def test_normalization(self):
        # Test that standardizing the inputs has zero effect on the braintreee model predictions
        # (prior to the neural model training, of course).
        model_raw = braintree.BrainTree(num_trees=15, standardize=False, train_steps=0,
                                        subsample=1.0)
        model_raw.fit(concrete_train, concrete_test)
        scores_raw = model_raw.predict(concrete_test)
        model_norm = braintree.BrainTree(num_trees=15, train_steps=0, subsample=1.0)
        model_norm.fit(concrete_train, concrete_test)
        scores_norm = model_norm.predict(concrete_test)
        np.testing.assert_array_almost_equal(scores_raw, scores_norm)
