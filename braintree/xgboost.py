"""Thin wrappers around XGBoost classes.

There are three reasons for these wrappers, instead of just directly calling XGBoost in the
rest of the codebase. First, these wrappers fix a few quirks of the XGBoost API design that
have been bugging me for a while. Second, these classes support additional functionality
(most notably, weight matrix extractors). Finally, these classes support a broader spectrum
of models than vanilla XGBoost (like multiple responses and multivariate distributions).
"""

import os
import re

import numpy as np
import xgboost as xgb


def _nans(*shape):
    """Creates a NumPy array of a given shape filled with NaNs."""
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
    TREE_DUMP_FILENAME = ".tree_dump.tmp"
    # Regular expressions for parsing XGBoost model dumps
    LEADING_TABS = re.compile(r"[^\t]")
    INTEGER = r"[0-9]+"
    FLOAT = r"-?[0-9]+(\.[0-9]+)?"
    SPLIT = re.compile(r"f(?P<predictor>" + INTEGER + ")<(?P<bias>" + FLOAT + ")")
    TERMINAL = re.compile(r"leaf=(?P<bias>" + FLOAT + ")")

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
        """Initializes split matrices with appropriate dimensions and values."""
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
        """Fits the model.

        Returns:
            xgbModel: The fitted model.
        """
        evaluation_pairs = [(self.train_data, "train"),
                            (self.validation_data, "validation")]
        training_parameters = {"seed": seed,
                               "max_depth": self.max_depth}
        self.model = xgb.train(training_parameters, self.train_data, self.num_trees,
                               evals=evaluation_pairs)
        self._parse_parameters()
        return self

    def _parse_parameters(self):
        """Parses the parameters of an XGBoost model from a dump of the tree lines."""
        model_dump = self._get_model_dump()
        trees = self._split_trees(model_dump)
        for i, tree in enumerate(trees):
            self._parse_tree(tree, i)

    def _get_model_dump(self):
        """Returns a list of lines from the text dump of an XGBoost model."""
        self.model.dump_model(self.TREE_DUMP_FILENAME)
        with open(self.TREE_DUMP_FILENAME, "r") as infile:
            dump_lines = infile.read().split("\n")[:-1]
        os.remove(self.TREE_DUMP_FILENAME)
        return dump_lines

    @staticmethod
    def _split_trees(tree_lines):
        """Splits a list of text lines from a tree dump into separate trees."""
        trees = []
        current_tree = []
        for line in tree_lines[1:]:
            if line.find("booster") == 0:
                trees += [current_tree]
                current_tree = []
            else:
                current_tree += [line]
        trees += [current_tree]
        return trees

    def _parse_tree(self, tree, tree_ndx):
        """Parses the parameters from a single tree."""
        terminal_ndx = 0
        for line in tree:
            depth = self.LEADING_TABS.search(line).start()
            split_match = self.SPLIT.search(line)
            if not split_match:
                terminal_ndx = self._parse_terminal(line, tree_ndx, terminal_ndx, depth)

    def _parse_terminal(self, line, tree_ndx, terminal_ndx, depth):
        """Parses a single terminal node of a single tree."""
        terminal_match = self.TERMINAL.search(line)
        terminal_bias = float(terminal_match.group("bias"))
        depth_from_max = self.max_depth - depth
        for _ in range(2 ** depth_from_max):
            self.terminal_bias[terminal_ndx, 0, tree_ndx] = terminal_bias
            terminal_ndx += 1
        return terminal_ndx
