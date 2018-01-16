import unittest

from context import data
from context import xgboost
from context import concrete

concrete_data = data.BrainTreeData(concrete[:, :7], concrete[:, 7:])
concrete_train, concrete_test = concrete_data.split(0.7)


class FitTest(unittest.TestCase):
    def test_model_fitting(self):
        model = xgboost.XgbModel(concrete_train.to_dmatrix(), concrete_test.to_dmatrix()).fit()
