import pandas as pd
import numpy as np

from . import tree


class NotFitException(Exception):
    pass


class Boruta(object):
    def __init__(self, num_noise_columns, num_rounds, threshold_quantile=0.8):
        self.num_noise_columns = num_noise_columns
        self.num_rounds = num_rounds
        self.threshold_quantile = threshold_quantile
        self.fit_results = None

    @property
    def cutoff_score(self):
        if self.fit_results is None:
            return NotFitException("Results must be fit first!")
        noise_importance = self.fit_results.loc[self.fit_results["noise"], "importance"]
        return np.percentile(noise_importance, 100 * self.threshold_quantile)

    @property
    def average_importance(self):
        if self.fit_results is None:
            return NotFitException("Results must be fit first!")
        return (self.fit_results
                .loc[~self.fit_results["noise"], :]
                .groupby(["variable_ndx"])
                .agg({"importance": "mean"}))

    @property
    def recommended_drops(self):
        if self.fit_results is None:
            return NotFitException("Results must be fit first!")
        below_cutoff = self.average_importance["importance"] < self.cutoff_score
        return self.average_importance.index[below_cutoff].tolist()

    def fit(self, data, **tree_params):
        results_list = [self.fit_round(data, tree_params).assign(iteration=iteration)
                        for iteration in range(self.num_rounds)]
        self.fit_results = pd.concat(results_list)

    def fit_round(self, data, tree_params):
        noised_data = data.add_noise_columns(self.num_noise_columns)
        train, test = noised_data.shuffle().split(0.7)
        model = tree.TreeModel(**tree_params)
        model.fit(train.to_dmatrix(), test.to_dmatrix())
        results = pd.DataFrame({
            "importance": model.importance,
            "noise": False,
            "variable_ndx": range(noised_data.num_features)
        })
        results.iloc[-self.num_noise_columns:, 1] = True
        return results
