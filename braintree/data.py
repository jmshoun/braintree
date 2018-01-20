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

    def split(self, split_fraction, seed=0):
        """Splits a BrainTreeData object into two disjoint data sets.

        Args:
            split_fraction (float): The fraction of data to place in the first of the two
                data sets.
            seed (int): Seed for the random number generator.
        Returns:
            (BrainTreeData, BrainTreeData): Two data sets with the original data randomly
                distributed between them.
        """
        if split_fraction <= 0 or split_fraction >= 1:
            raise ValueError("split_fraction must be between 0 and 1.")
        split_row = int(self.predictors.shape[0] * split_fraction)
        if split_row == 0:
            raise ValueError("split_value is too extreme; one data set is empty.")
        np.random.seed(seed)
        np.random.shuffle(self.predictors)
        # Second call to seed to ensure permutation for predictors and responses is the same.
        np.random.seed(seed)
        np.random.shuffle(self.responses)
        return (BrainTreeData(self.predictors[:split_row, :], self.responses[:split_row, :]),
                BrainTreeData(self.predictors[split_row:, :], self.responses[split_row:, :]))

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
