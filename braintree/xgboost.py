"""Thin wrappers around XGBoost classes.

There are three reasons for these wrappers, instead of just directly calling XGBoost in the
rest of the codebase. First, these wrappers fix a few quirks of the XGBoost API design that
have been bugging me for a while. Second, these classes support additional functionality
(most notably, weight matrix extractors). Finally, these classes support a broader spectrum
of models than vanilla XGBoost (like multiple responses and multivariate distributions).
"""

import numpy as np
import xgboost as xgb


def _nans(*shape):
    """Create a NumPy array of a given shape filled with NaNs."""
    return np.full(shape, np.nan)


class XgbModel(object):
    """Wrapper around a basic XGBoost model.

    Attributes:
        train_data (xgb.DMatrix): Training data for the model.
        validation_data (xgb.DMatrix): Validation data for the model.
        num_trees (int): Number of trees in the XGBoost model.
        max_depth (int): Maximum depth of each tree in the model.
        model (xgb.Booster): Fitted XGBoost model.
        terminal_bias (numpy.ndarray): Biases (constants) for values of the terminal nodes.
        split_weight (list[numpy.ndarray]): Weights for tree splits.
        split_bias (list[numpy.ndarray]): Biases for tree splits.
        split_strength (list[numpy.ndarray]): Strengths of tree splits.
    """

    def __init__(self, train_data, validation_data=None, num_trees=50, max_depth=5):
        """Default constructor."""
        self.train_data = train_data
        self.validation_data = validation_data
        self.num_trees = num_trees
        self.max_depth = max_depth
        self.model = None
        # Placeholders for extracted model parameters.
        num_terminal_nodes = 2 ** self.max_depth
        self.terminal_bias = _nans(num_terminal_nodes, 1, self.num_trees)
        self.split_weight, self.split_bias, self.split_strength = self._initialize_splits()

    def _initialize_splits(self):
        """Initialize split matrices with appropriate dimensions and values."""
        num_features = self.train_data.num_col()
        # There is one set of split weighs for each layer of each tree; thus the matrices.
        split_weight = []
        split_bias = []
        split_strength = []
        for depth in range(self.max_depth):
            split_weight += [np.zeros((2 ** depth, num_features, self.num_trees))]
            split_bias += [_nans(2 ** depth, 1, self.num_trees)]
            split_strength += [_nans(2 ** depth, 1, self.num_trees)]
        return split_weight, split_bias, split_strength

    def fit(self, seed=0):
        """Fit the model.

        Returns:
            xgbModel: The fitted model.
        """
        evaluation_pairs = [(self.train_data, "train"),
                            (self.validation_data, "validation")]
        training_parameters = {"seed": seed,
                               "max_depth": self.max_depth}
        self.model = xgb.train(training_parameters, self.train_data, self.num_trees,
                               evals=evaluation_pairs)
        return self
