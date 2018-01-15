"""BrainTree data class.

Many machine learning libraries require custom data formats. Unfortunately, this library is
no exception. This is largely because BrainTree is built on top of XGBoost and TensorFlow,
each of which has its own arcane data format requirements. The BrainTree data format exists
primarily to abstract away these difficulties.
"""

import numpy as np
import xgboost as xgb


class BrainTreeData(object):
    """Data to be fed into a BrainTree model.
    
    Attributes:
        predictors (numpy.ndarray): n by k numeric matrix of predictors.
        responses (numpy.ndarray): n by m numeric matrix of responses.
    """
    def __init__(self, predictors, responses):
        """Default constructor."""
        self.predictors = predictors
        self.responses = responses

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
