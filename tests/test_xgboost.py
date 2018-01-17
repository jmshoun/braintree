import unittest

from context import data
from context import xgboost
from context import concrete

concrete_data = data.BrainTreeData(concrete[:, :7], concrete[:, 7:])
concrete_train, concrete_test = concrete_data.split(0.7)


class FitTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = xgboost.XgbModel(concrete_train.to_dmatrix(), concrete_test.to_dmatrix(),
                                     max_depth=4, num_trees=25)
        cls.model.fit()

    def test_parameter_matrix_dimensions(self):
        terminal_bias_shape = list(self.model.terminal_bias.shape)
        self.assertListEqual(terminal_bias_shape, [16, 1, 25])
        split_weight_shape_3 = list(self.model.split_weight[3].shape)
        self.assertListEqual(split_weight_shape_3, [8, 7, 25])
        split_bias_shape_2 = list(self.model.split_bias[2].shape)
        self.assertListEqual(split_bias_shape_2, [4, 1, 25])
        split_strength_0 = list(self.model.split_strength[0].shape)
        self.assertListEqual(split_strength_0, [1, 1, 25])
