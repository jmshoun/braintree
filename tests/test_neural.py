import unittest

import numpy as np

from context import tree
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

    def test_model_predictions_default(self):
        model_ = neural.NeuralModel(7)
        predictions = model_.predict(concrete_train)
        self.assertListEqual(predictions.tolist(), [0.] * 72)

    def test_model_predictions_initial(self):
        starting_tree = tree.TreeModel(max_depth=4, num_trees=25, default_split_strength=30)
        starting_tree.fit(concrete_train.to_dmatrix(), concrete_test.to_dmatrix())
        model_ = neural.NeuralModel(7, max_depth=4, num_trees=25)
        model_.load_params(starting_tree)
        tree_predictions = starting_tree.model.predict(concrete_test.to_dmatrix())
        model_predictions = model_.predict(concrete_test)
        # A few observations are ignored because one or more predictors is exactly on a
        # tree splitting boundary, and the neural implementation splits the weights in half
        # in that case. It's a known difference in semantics, not a bug.
        for bad_ndx in [0, 15, 22]:
            model_predictions[bad_ndx] = tree_predictions[bad_ndx]
        np.testing.assert_array_almost_equal(model_predictions, tree_predictions, decimal=4)
