"""BrainTree data class.

Many machine learning libraries require custom data formats. Unfortunately, this library is
no exception. This is largely because BrainTree is built on top of XGBoost and TensorFlow,
each of which has its own arcane data format requirements. The BrainTree data format exists
primarily to abstract away these difficulties.
"""

import copy

import xgboost as xgb
import numpy as np
import pandas as pd


class BrainTreeData(object):
    """Data to be fed into a BrainTree model.

    Attributes:
        predictors (numpy.ndarray): n by k numeric matrix of predictors.
        responses (numpy.ndarray): n by m numeric matrix of responses.
        standard_factors (dict): Dictionary of factors to standardize raw data by columns.
        standardized (bool): Whether the data has been standardized.
    """
    def __init__(self, predictors, responses, predictor_names=None, response_names=None):
        """Default constructor."""
        self.predictors = self._force_2d(predictors.copy())
        self.responses = self._force_2d(responses.copy())
        self.predictor_names = predictor_names
        self.response_names = response_names
        self._assert_predictors_and_responses_same_size()
        self._assert_correct_name_length()
        self.standard_factors = {}
        self.standardized = False

    @classmethod
    def from_csv(cls, filename, response_names, feature_names=None, header=False, delimiter=","):
        """Constructor from a CSV file."""
        if not header and feature_names is None:
            raise ValueError("header must be true or feature_names must not be None!")
        pandas_header = 0 if header else None
        df = pd.read_csv(filename, delimiter=delimiter, header=pandas_header)
        if feature_names is not None:
            df.columns = feature_names
        return cls.from_data_frame(df, response_names)

    @classmethod
    def from_data_frame(cls, df, response_names):
        """Constructor from a Pandas DataFrame."""
        feature_names = df.columns.values
        predictor_names = list(set(feature_names) - set(response_names))
        predictors = df.loc[:, predictor_names].values
        responses = df.loc[:, response_names].values
        return cls(predictors, responses, predictor_names, response_names)

    @property
    def num_observations(self):
        """The number of observations in the data set."""
        return self.predictors.shape[0]

    @property''
    def num_features(self):
        """The number of features (predictors) in the data set."""
        return self.predictors.shape[1]

    def drop_columns(self, columns):
        """Drop specified columns from a data set.
        
        Args:
            columns (list): List of predictor columns to drop. Can be a mix of integer indices
                and strings of column names.
        """
        column_indices = []
        for column in columns:
            if isinstance(column, int):
                if column >= self.num_features:
                    raise ValueError(f"Can't delete column #{column}; "
                                     + f"data only has {self.num_features} columns!")
                column_indices += [column]
            else:
                ndx_lookup = [(i, name) for i, name in enumerate(self.predictor_names)
                              if name == column]
                if len(ndx_lookup) == 0:
                    raise ValueError(f"Provided column name {column} is not in the data set!")
                column_indices += [ndx_lookup[0][0]]
        self.predictors = np.delete(self.predictors, column_indices, axis=1)
        return self

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
        self.standardized = True

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

    def _assert_correct_name_length(self):
        """Ensures the length of predictor_names and response_names are right."""
        if self.predictor_names is not None and len(self.predictor_names) != self.num_features:
            raise ValueError("predictor_names must be the same length as the number of predictors!")
        if self.response_names is not None and len(self.response_names) != self.responses.shape[1]:
            raise ValueError("response_names must be the same length as the number of responses!")

    def add_noise_columns(self, num_columns):
        """Adds some columns of uncorrelated Gaussian noise to the predictors.
        
        Args:
            num_columns (int): Number of columns to add to the data set.
        """
        noise = np.random.normal(0, 1, [self.num_observations, num_columns])
        new_predictors = np.concatenate([self.predictors, noise], axis=1)
        return BrainTreeData(new_predictors, self.responses)

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
        return (BrainTreeData(self.predictors[:split_row, :], self.responses[:split_row, :],
                              self.predictor_names, self.response_names),
                BrainTreeData(self.predictors[split_row:, :], self.responses[split_row:, :],
                              self.predictor_names, self.response_names))

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
        return self

    def to_array_generator(self, batch_size, all_=True, repeat=False, response_number=0):
        """Creates a generator that iterates over the data set and yields batch-sized ndarrays.

        Args:
            batch_size (int): The size of each batch.
            all_ (bool): If True, return every observation. Otherwise, return the most observations
                that are cleanly divisble by the batch size.
            repeat (bool): If True, continue looping through the data set ad infinitum.
            response_number (int): The index of the response to return.
        Returns:
            Generator of (predictor, response) ndarray pairs.
        """
        ndx = 0
        num_observations = self.predictors.shape[0]
        num_features = self.predictors.shape[1]
        new_predictor_shape = [1, batch_size, num_features]
        # Main loop through the data
        while True:
            if ndx + batch_size <= num_observations:
                yield (self.predictors[ndx:(ndx + batch_size), :].reshape(new_predictor_shape),
                       self.responses[ndx:(ndx + batch_size), response_number])
            elif all_:
                # Annoying edge case when num_observations % batch_size != 0
                remainder_shape = [1, num_observations - ndx, num_features]
                num_fills = batch_size - (num_observations - ndx)
                fill_shape = [1, num_fills, num_features]
                yield (np.hstack([self.predictors[ndx:, :].reshape(remainder_shape),
                                  self.predictors[:num_fills, :].reshape(fill_shape)]),
                       np.hstack([self.responses[ndx:, response_number],
                                  self.responses[:num_fills, response_number]]))
            # Increment ndx and check for halting conditions
            ndx += batch_size
            if ndx >= num_observations and not repeat:
                break
            else:
                ndx %= num_observations

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
