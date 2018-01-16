"""Thin wrappers around XGBoost classes.

There are three reasons for these wrappers, instead of just directly calling XGBoost in the
rest of the codebase. First, these wrappers fix a few quirks of the XGBoost API design that
have been bugging me for a while. Second, these classes support additional functionality
(most notably, weight matrix extractors). Finally, these classes support a broader spectrum
of models than vanilla XGBoost (like multiple responses and multivariate distributions).
"""

import xgboost as xgb


class XgbModel(object):
    """Wrapper around a basic XGBoost model.

    Attributes:
        train_data (xgb.DMatrix): Training data for the model.
        validation_data (xgb.DMatrix): Validation data for the model.
        num_trees (int): Number of trees in the XGBoost model.
        model (xgb.Booster): Fitted XGBoost model.
    """

    def __init__(self, train_data, validation_data=None, num_trees=50):
        """Default constructor."""
        self.train_data = train_data
        self.validation_data = validation_data
        self.num_trees = num_trees
        self.model = None

    def fit(self):
        """Fit the model.

        Returns:
            xgbModel: The fitted model.
        """
        evaluation_pairs = [(self.train_data, "train"),
                            (self.validation_data, "validation")]
        self.model = xgb.train({}, self.train_data, self.num_trees, evals=evaluation_pairs)
        return self
