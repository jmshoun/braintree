"""BrainTree data class.

Many machine learning libraries require custom data formats. Unfortunately, this library is
no exception. This is largely because BrainTree is built on top of XGBoost and TensorFlow,
each of which has its own arcane data format requirements. The BrainTree data format exists
primarily to abstract away these difficulties.
"""

import copy

import xgboost as xgb
import numpy as np


class BrainTreeData(object):
    """Data to be fed into a BrainTree model.

    Attributes:
        predictors (numpy.ndarray): n by k numeric matrix of predictors.
        responses (numpy.ndarray): n by m numeric matrix of responses.
        standard_factors (dict): Dictionary of factors to standardize raw data by columns.
    """
    def __init__(self, predictors, responses):
        """Default constructor."""
        self.predictors = self._force_2d(predictors)
        self.responses = self._force_2d(responses)
        self._assert_predictors_and_responses_same_size()
        self.standard_factors = {}

    @property
    def num_observations(self):
        """The number of observations in the data set."""
        return self.predictors.shape[0]

    @property
    def num_features(self):
        """The number of features (predictors) in the data set."""
        return self.predictors.shape[1]

    def standardize(self, standard_factors=None):
        """Standardize the data so it has zero mean and unit variance.

        Args:
            standard_factors (dict): If provided, use these factors for standardization
                instead of computing the factors from the data itself.
        """
        if standard_factors is not None:
            self.standard_factors = copy.deepcopy(standard_factors)
        else:
            self.standard_factors = self._compute_standard_factors()
        self.predictors = ((self.predictors - self.standard_factors["predictor_means"])
                           / self.standard_factors["predictor_sds"])
        self.responses = ((self.responses - self.standard_factors["response_means"])
                          / self.standard_factors["response_sds"])

    def _compute_standard_factors(self):
        """Compute standardization factors from the data itself."""
        return {"predictor_means": np.mean(self.predictors, axis=0),
                "predictor_sds": np.std(self.predictors, axis=0),
                "response_means": np.mean(self.responses, axis=0),
                "response_sds": np.std(self.responses, axis=0)}

    @staticmethod
    def _force_2d(array):
        """Forces an input array to be 2-dimensional."""
        shape = array.shape
        if len(shape) == 1:
            array.shape = (shape[0], 1)
        elif len(shape) > 2:
            raise ValueError("predictors and responses may not have more than 2 dimensions.")
        return array

    def _assert_predictors_and_responses_same_size(self):
        """Ensures the predictors and responses have compatible dimensions."""
        predictor_rows = self.predictors.shape[0]
        response_rows = self.responses.shape[0]
        if predictor_rows != response_rows:
            raise ValueError("predictors and responses must have the same number of rows.")

    def split(self, split_fraction):
        """Splits a BrainTreeData object into two disjoint data sets.

        Args:
            split_fraction (float): The fraction of data to place in the first of the two
                data sets.
        Returns:
            (BrainTreeData, BrainTreeData): Two data sets with the original data split between them.
        """
        if split_fraction <= 0 or split_fraction >= 1:
            raise ValueError("split_fraction must be between 0 and 1.")
        split_row = int(self.predictors.shape[0] * split_fraction)
        if split_row == 0:
            raise ValueError("split_value is too extreme; one data set is empty.")
        return (BrainTreeData(self.predictors[:split_row, :], self.responses[:split_row, :]),
                BrainTreeData(self.predictors[split_row:, :], self.responses[split_row:, :]))

    def shuffle(self, seed=0):
        """Randomly shuffles the order of the observations in the data set.

        Args:
            seed (int): Seed to pass to the NumPy random number generator.
        """
        np.random.seed(seed)
        np.random.shuffle(self.predictors)
        # Second call to seed to ensure permutation for predictors and responses is the same.
        np.random.seed(seed)
        np.random.shuffle(self.responses)

    def to_array_generator(self, batch_size, all_=True, response_number=0):
        """Creates a generator that iterates over the data set and yields batch-sized ndarrays.

        Args:
            batch_size (int): The size of each batch.
            all_ (bool): If True, return every observation. Otherwise, return the most observations
                that are cleanly divisble by the batch size.
            response_number (int): The index of the response to return.
        Returns:
            Generator of (predictor, response) ndarray pairs.
        """
        ndx = 0
        num_observations = self.predictors.shape[0]
        num_features = self.predictors.shape[1]
        new_predictor_shape = [1, batch_size, num_features]
        while ndx + batch_size <= num_observations:
            yield (self.predictors[ndx:(ndx + batch_size), :].reshape(new_predictor_shape),
                   self.responses[ndx:(ndx + batch_size), response_number])
            ndx += batch_size
        if ndx < num_observations and all_:
            remainder_shape = [1, num_observations % batch_size, num_features]
            num_fills = batch_size - (num_observations % batch_size)
            fill_predictors = np.zeros((1, num_fills, num_features))
            fill_responses = np.zeros((num_fills, ))
            yield (np.hstack([self.predictors[ndx:, :].reshape(remainder_shape), fill_predictors]),
                   np.hstack([self.responses[ndx:, response_number], fill_responses]))

    def to_dmatrix(self, response_number=0):
        """Creates an XGBoost DMatrix representation of the data.

        Args:
            response_number (int): The index of the column in responses to use as the
                response. DMatrix only supports a single response value per observation.
        Returns:
            xgb.DMatrix: The data with the given response attached.
        """
        num_responses = self.responses.shape[1]
        if response_number < 0 or response_number >= num_responses:
            raise ValueError("response_number must be between 0 and {}.".format(num_responses))
        response = self.responses[:, response_number]
        return xgb.DMatrix(self.predictors, label=response)
