import unittest

import numpy as np

from context import data


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


class TestInputValidation(unittest.TestCase):
    """Test the runtime input validation performed in the constructor."""
    def test_2d_coercion(self):
        # 1-D inputs should be coerced to 2-D.
        data_ = data.BrainTreeData(np.zeros(20), np.ones(20))
        self.assertListEqual(list(data_.predictors.shape), [20, 1])
        self.assertListEqual(list(data_.responses.shape), [20, 1])

    def test_3d_input(self):
        # #-D Input in either position should be rejected out of hand.
        with self.assertRaises(ValueError):
            data_ = data.BrainTreeData(np.zeros((3, 2, 5)), np.zeros(5))
        with self.assertRaises(ValueError):
            data_ = data.BrainTreeData(np.zeros((3, 2)), np.zeros((5, 3, 7)))

    def test_mismatched_lengths(self):
        with self.assertRaises(ValueError):
            data_ = data.BrainTreeData(np.zeros(10), np.zeros(15))
