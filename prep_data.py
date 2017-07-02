import pickle

import sklearn.model_selection
import sklearn.preprocessing
import numpy as np


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

print("Loading data...")
songs = load_data("data/YearPredictionMSD.txt")
print("Standardizing data...")
songs = standardize_data(songs)
print("Pickling data...")
with open("data/song_matrix.p", "wb") as outfile:
    pickle.dump(songs, outfile, pickle.HIGHEST_PROTOCOL)
