"""BrainTree data class.

Many machine learning libraries require custom data formats. Unfortunately, this library is
no exception. This is largely because BrainTree is built on top of XGBoost and TensorFlow,
each of which has its own arcane data format requirements. The BrainTree data format exists
primarily to abstract away these difficulties.
"""

import xgboost as xgb


class BrainTreeData(object):
    """Data to be fed into a BrainTree model.

    Attributes:
        predictors (numpy.ndarray): n by k numeric matrix of predictors.
        responses (numpy.ndarray): n by m numeric matrix of responses.
    """
    def __init__(self, predictors, responses):
        """Default constructor."""
        self.predictors = self._force_2d(predictors)
        self.responses = self._force_2d(responses)
        self._assert_predictors_and_responses_same_size()

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

    def to_dmatrix(self, response_number=0):
        """Creates an XGBoost DMatrix representation of the data.

        Args:
            response_number (int): The index of the column in responses to use as the
                response. DMatrix only supports a single response value per observation.
        Returns:
            xgb.DMatrix: The data with the given response attached.
        """
        response = self.responses[:, response_number]
        return xgb.DMatrix(self.predictors, label=response)
