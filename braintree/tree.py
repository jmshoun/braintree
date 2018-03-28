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


def _zeros(*shape):
    """Creates a NumPy array of a given shape filled with NaNs."""
    return np.zeros(shape)


class NotFitError(Exception):
    pass


class TreeModel(object):
    """Wrapper around a basic XGBoost model.

    Attributes:
        num_trees (int): Number of trees in the XGBoost model.
        max_depth (int): Maximum depth of each tree in the model.
        subsample (float): Subsampling rate when training the model.
        default_split_strength (float): Default value of split strength parameters.
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
    FLOAT = r"-?[0-9]+(\.[0-9]+)?(e[+-]?[0-9]{2})?"
    SPLIT = re.compile(r"f(?P<predictor>" + INTEGER + ")<(?P<bias>" + FLOAT + ")")
    TERMINAL = re.compile(r"leaf=(?P<bias>" + FLOAT + ")")

    def __init__(self, num_trees=50, max_depth=5, subsample=1.0, default_split_strength=2.0,
                 eta=0.1, min_leaf_weight=1.0, column_subsample=1.0):
        """Default constructor."""
        self.num_trees = num_trees
        self.max_depth = max_depth
        self.subsample = subsample
        self.default_split_strength = default_split_strength
        self.eta = eta
        self.column_subsample = column_subsample
        self.min_leaf_weight = min_leaf_weight
        self.model = None
        # Placeholders for extracted model parameters.
        num_terminal_nodes = 2 ** self.max_depth
        self.terminal_bias = _zeros(num_terminal_nodes, 1, self.num_trees)
        self.num_predictors = None
        self.split_weight, self.split_bias, self.split_strength = None, None, None

    def _initialize_splits(self):
        """Initializes split matrices with appropriate dimensions and values."""
        # There is one set of split weighs for each layer of each tree; thus the matrices.
        self.split_weight = []
        self.split_bias = []
        self.split_strength = []
        for depth in range(self.max_depth):
            self.split_weight += [_zeros(2 ** depth, self.num_predictors, self.num_trees)]
            self.split_bias += [_zeros(2 ** depth, 1, self.num_trees)]
            self.split_strength += [_zeros(2 ** depth, 1, self.num_trees)]

    def fit(self, train_data, validation_data, seed=0):
        """Fits the model.

        Returns:
            xgbModel: The fitted model.
        """
        evaluation_pairs = [(train_data, "train"),
                            (validation_data, "validation")]
        training_parameters = {"seed": seed,
                               "max_depth": self.max_depth,
                               "subsample": self.subsample,
                               "eta": self.eta,
                               "min_child_weight": self.min_leaf_weight,
                               "colsample_bylevel": self.column_subsample,
                               "silent": 1,
                               "base_score": 0.0}
        self.model = xgb.train(training_parameters, train_data, self.num_trees,
                               evals=evaluation_pairs, verbose_eval=False)
        self.num_predictors = train_data.num_col()
        self._initialize_splits()
        self._parse_parameters()
        return self

    @property
    def importance(self):
        if not self.model:
            raise NotFitError("The tree model has not been fit yet!")
        importance_dict = self.model.get_score(importance_type="gain")
        importance = np.zeros(self.num_predictors)
        for var_name, imp in importance_dict.items():
            var_ndx = int(var_name[1:])
            importance[var_ndx] = imp
        return importance / np.sum(importance)

    def add_noise(self, split_weight_noise, split_bias_noise):
        split_weight_sd = split_weight_noise / np.sqrt(self.num_predictors)
        for depth in range(self.max_depth):
            self.split_weight[depth] += np.random.normal(scale=split_weight_sd,
                                                         size=self.split_weight[depth].shape)
            self.split_bias[depth] += np.random.normal(scale=split_bias_noise,
                                                       size=self.split_bias[depth].shape)

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
        # Keep track of the current index of the terminal node.
        terminal_ndx = 0
        for line in tree:
            depth = self.LEADING_TABS.search(line).start()
            split_match = self.SPLIT.search(line)
            if split_match:
                self._parse_split(split_match, tree_ndx, terminal_ndx, depth)
            else:
                terminal_ndx = self._parse_terminal(line, tree_ndx, terminal_ndx, depth)

    def _parse_split(self, split_match, tree_ndx, terminal_ndx, depth):
        """Parses a single split of a single tree."""
        split_variable_ndx = int(split_match.group("predictor"))
        split_bias = float(split_match.group("bias"))
        depth_from_max = self.max_depth - depth
        split_ndx = terminal_ndx // (2 ** depth_from_max)
        self.split_bias[depth][split_ndx, 0, tree_ndx] = split_bias
        self.split_weight[depth][split_ndx, split_variable_ndx, tree_ndx] = -1
        self.split_strength[depth][split_ndx, 0, tree_ndx] = self.default_split_strength

    def _parse_terminal(self, line, tree_ndx, terminal_ndx, depth):
        """Parses a single terminal node of a single tree."""
        terminal_match = self.TERMINAL.search(line)
        terminal_bias = float(terminal_match.group("bias"))
        # If the terminal node isn't at the maximum depth of the tree, then the terminal
        # will correspond to multiple elements of the terminal bias matrix, since the
        # terminal bias matrix is fixed-depth.
        depth_from_max = self.max_depth - depth
        for _ in range(2 ** depth_from_max):
            self.terminal_bias[terminal_ndx, 0, tree_ndx] = terminal_bias
            terminal_ndx += 1
        return terminal_ndx
