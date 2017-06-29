import re
import os

import xgboost as xgb
import numpy as np
import sklearn.model_selection
import sklearn.preprocessing


def load_data(filename, response_column=0):
    """Loads a CSV data set and returns NumPy train, validation and test data sets."""
    data = np.genfromtxt(filename, delimiter=",")
    y = data[:, response_column, None]
    X = np.delete(data, response_column, axis=1)
    X_train, X_nontrain, y_train, y_nontrain = sklearn.model_selection.train_test_split(
        X, y, test_size=0.4, random_state=1729)
    X_validation, X_test, y_validation, y_test = sklearn.model_selection.train_test_split(
        X_nontrain, y_nontrain, test_size=0.4, random_state=1729)
    return {"train": {"X": X_train, "y": y_train},
            "validation": {"X": X_validation, "y": y_validation},
            "test": {"X": X_test, "y": y_test}}


def standardize_data(data):
    """Standardizes the predictors and response to have zero mean and unit variance."""
    x_standardizer = sklearn.preprocessing.StandardScaler()
    x_standardizer.fit(data["train"]["X"])
    y_standardizer = sklearn.preprocessing.StandardScaler()
    y_standardizer.fit(data["train"]["y"])

    def standardize(data_set):
        return {"X": x_standardizer.transform(data_set["X"]),
                "y": y_standardizer.transform(data_set["y"])}

    return {k: standardize(v) for k, v in data.items()}


def convert_to_dmatrix(data):
    return xgb.DMatrix(data["X"], label=data["y"])


def fit_gbm(data):
    """Fits a GBM to training data."""
    xgb_data = {k: convert_to_dmatrix(v) for k, v in data.items()}
    return xgb.train({"eta": 0.3, "max_depth": 6},
                      dtrain=xgb_data["train"],
                      num_boost_round=100,
                      evals=[(xgb_data["train"], "train"), (xgb_data["validation"], "validation")],
                      verbose_eval=False)


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


songs = load_data("data/YearPredictionMSD.txt")
songs = standardize_data(songs)
num_predictors = songs["train"]["X"].shape[1]

song_model = fit_gbm(songs)
song_params = gbm_to_params(song_model, num_predictors)
