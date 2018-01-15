import unittest

import numpy as np

from .context import data


class TestXgboost(unittest.TestCase):
    """Test the XGBoost DMatrix representation of the data."""
    @classmethod
    def setUpClass(cls):
        cls.predictors = np.zeros((10, 4))
        cls.responses = np.zeros((10, 2))
        cls.data = data.BrainTreeData(cls.predictors, cls.responses)
        cls.xgb_data = cls.data.to_dmatrix(1)

    def test_output_dimensions(self):
        self.assertEqual(self.xgb_data.num_row(), 10)
        self.assertEqual(self.xgb_data.num_col(), 4)

    def test_output_contents(self):
        np.testing.assert_array_equal(self.xgb_data.get_label(), self.responses[:, 1])
