import unittest

import numpy as np

from context import data
from context import concrete

class TestData(unittest.TestCase):
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


class TestDataSplit(unittest.TestCase):
    """Test the split method of BrainTreeData."""
    def test_split_dimensions(self):
        data_ = data.BrainTreeData(np.zeros(10), np.ones(10))
        train, test = data_.split(0.7)
        self.assertEqual(train.predictors.shape[0], 7)
        self.assertEqual(test.predictors.shape[0], 3)

    def test_split_matches(self):
        # Ensure that predictors and responses are still correctly paired after the split
        data_ = data.BrainTreeData(np.arange(15), np.arange(15))
        train, test = data_.split(0.6)
        self.assertListEqual(train.predictors.tolist(), train.responses.tolist())
        self.assertListEqual(test.predictors.tolist(), test.responses.tolist())

    def test_split_fraction_validation(self):
        # Ensure that the validation on split_fraction works as planned.
        data_ = data.BrainTreeData(np.zeros(10), np.zeros(10))
        with self.assertRaises(ValueError):
            data_.split(-3)
        with self.assertRaises(ValueError):
            data_.split(1.1)
        with self.assertRaises(ValueError):
            # This yields zero rows in the first data set, and so is disallowed.
            data_.split(0.08)


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
            data.BrainTreeData(np.zeros((3, 2, 5)), np.zeros(5))
        with self.assertRaises(ValueError):
            data.BrainTreeData(np.zeros((3, 2)), np.zeros((5, 3, 7)))

    def test_mismatched_lengths(self):
        with self.assertRaises(ValueError):
            data.BrainTreeData(np.zeros(10), np.zeros(15))

    def test_response_column(self):
        data_ = data.BrainTreeData(np.zeros((25, 3)), np.zeros((25, 2)))
        data_.to_dmatrix(1)
        with self.assertRaises(ValueError):
            data_.to_dmatrix(-1)
        with self.assertRaises(ValueError):
            data_.to_dmatrix(2)


class TestDataStandardization(unittest.TestCase):
    def test_default_standardization(self):
        data_ = data.BrainTreeData(concrete[:, :7], concrete[:, 7:])
        data_.standardize()
        np.testing.assert_almost_equal(np.mean(data_.predictors, axis=0), [0] * 7)
        np.testing.assert_almost_equal(np.mean(data_.responses, axis=0), [0] * 3)
        np.testing.assert_almost_equal(np.std(data_.predictors, axis=0), [1] * 7)
        np.testing.assert_almost_equal(np.std(data_.responses, axis=0), [1] * 3)

    def test_alternate_standardization(self):
        # Compute true means and sds for reference
        data_ = data.BrainTreeData(concrete[:, :7], concrete[:, 7:])
        predictor_means = np.mean(data_.predictors, axis=0)
        predictor_sds = np.std(data_.predictors, axis=0)
        response_means = np.mean(data_.responses, axis=0)
        response_sds = np.std(data_.responses, axis=0)
        # Standardize with forced values
        data_.standardize({"predictor_means": [10] * 7,
                           "predictor_sds": [5] * 7,
                           "response_means": [10] * 3,
                           "response_sds": [5] * 3})
        # Assert that the standardization used the supplied parameters
        np.testing.assert_almost_equal(np.mean(data_.predictors, axis=0),
                                       (predictor_means - 10) / 5)
        np.testing.assert_almost_equal(np.std(data_.predictors, axis=0), predictor_sds / 5)
        np.testing.assert_almost_equal(np.mean(data_.responses, axis=0), (response_means - 10) / 5)
        np.testing.assert_almost_equal(np.std(data_.responses, axis=0), response_sds / 5)
