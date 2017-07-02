import warnings
import re
import os
import pickle

import numpy as np

import braintree

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    import xgboost as xgb


def convert_to_dmatrix(data):
    return xgb.DMatrix(data["X"], label=data["y"])


def fit_gbm(data):
    """Fits a GBM to training data."""
    xgb_data = {k: convert_to_dmatrix(v) for k, v in data.items()}
    return xgb.train({"eta": 0.2, "max_depth": 6, "subsample": 0.5},
                      dtrain=xgb_data["train"],
                      num_boost_round=500,
                      evals=[(xgb_data["train"], "train"), (xgb_data["validation"], "validation")],
                      verbose_eval=True)


def gbm_to_params(model, num_predictors):
    tree_filename = "tree_dump.tmp"
    leading_tabs_re = re.compile(r"[^\t]")

    model.dump_model(tree_filename)
    with open(tree_filename, "r") as tree_file:
        tree_lines = tree_file.read().split("\n")[:-1]
    os.remove(tree_filename)

    max_depth = max([leading_tabs_re.search(line).start() for line in tree_lines])
    trees = split_trees(tree_lines)
    num_trees = len(trees)

    params = {"terminal_bias": np.zeros([num_trees, 2 ** max_depth]),
              "split_bias": [np.zeros([num_trees, 2 ** depth]) for depth in range(max_depth)],
              "split_strength": [np.zeros([num_trees, 2 ** depth]) for depth in range(max_depth)],
              "split_weight": [np.zeros([num_trees, 2 ** depth, num_predictors])
                               for depth in range(max_depth)]}

    for i, tree in enumerate(trees):
        params = parse_tree(tree, i, max_depth, params)
    return params


def split_trees(tree_lines):
    trees = []
    current_tree = []
    for line in tree_lines[1:]:
        if line.find("booster") == 0:
            trees.append(current_tree)
            current_tree = []
        else:
            current_tree.append(line)
    trees.append(current_tree)
    return trees


def parse_tree(tree, tree_number, max_depth, params):
    leading_tabs_re = re.compile(r"[^\t]")
    split_re = re.compile(r"f(?P<predictor>[0-9]+)<(?P<bias>-?[0-9]+(\.[0-9]+)?)")
    terminal_re = re.compile(r"leaf=(?P<bias>-?[0-9]+(\.[0-9]+)?)")
    current_index = 0
    for line in tree:
        depth = leading_tabs_re.search(line).start()
        split_match = split_re.search(line)
        if split_match:
            split_predictor = int(split_match.group("predictor"))
            split_bias = float(split_match.group("bias"))
            split_index = current_index // (2 ** (max_depth - depth))
            params["split_bias"][depth][tree_number, split_index] = split_bias
            params["split_weight"][depth][tree_number, split_index, split_predictor] = 1
            params["split_strength"][depth][tree_number, split_index] = 3
        else:
            terminal_match = terminal_re.search(line)
            terminal_bias = float(terminal_match.group("bias"))
            for _ in range(2 ** (max_depth - depth)):
                params["terminal_bias"][tree_number, current_index] = terminal_bias
                current_index += 1

    return params


with open("data/song_matrix.p", "rb") as infile:
    songs = pickle.load(infile)
num_predictors = songs["train"]["X"].shape[1]

train_data = braintree.TensorFlowData(songs["train"]["X"], songs["train"]["y"])
validation_data = braintree.TensorFlowData(songs["validation"]["X"], songs["validation"]["y"])
model = braintree.BrainTree(num_predictors, num_trees=50, max_depth=3,
                            batch_size=64, learning_rate=0.001)
model.train(train_data, validation_data, training_steps=10000, print_every=100)
