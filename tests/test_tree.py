import unittest

from context import data
from context import tree
from context import concrete

concrete_data = data.BrainTreeData(concrete[:, :7], concrete[:, 7:])
concrete_data.shuffle()
concrete_train, concrete_test = concrete_data.split(0.7)


class TreeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = tree.TreeModel(concrete_train.to_dmatrix(), concrete_test.to_dmatrix(),
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

    def test_terminal_bias_parsing(self):
        # All test values that follow were verified by manual inspection of tree dump.
        self.assertAlmostEqual(self.model.terminal_bias[0, 0, 0], 3.8625)
        # Parse negative values?
        self.assertAlmostEqual(self.model.terminal_bias[8, 0, 2], -0.040793)
        # Parse the last value?
        self.assertAlmostEqual(self.model.terminal_bias[15, 0, 24], 0.118218)

    def test_terminal_bias_assignment(self):
        for i in range(8, 12):
            self.assertAlmostEqual(self.model.terminal_bias[i, 0, 5], 1.72913)

    def test_split_bias_parsing(self):
        # Check several levels in the same tree
        self.assertAlmostEqual(self.model.split_bias[0][0, 0, 0], 126.5)
        self.assertAlmostEqual(self.model.split_bias[1][1, 0, 0], 187.5)
        self.assertAlmostEqual(self.model.split_bias[3][4, 0, 0], 190.5)
        # Parse last value?
        self.assertAlmostEqual(self.model.split_bias[3][3, 0, 24], 177.95)

    def test_split_bias_default_values(self):
        # This split doesn't appear in the tree, so should have a bias of 0.
        self.assertAlmostEqual(self.model.split_bias[2][3, 0, 4], 0.0)

    def test_split_weight_values(self):
        # Test the first split
        self.assertEqual(self.model.split_weight[0][0, 1, 0], -1)
        for i in [0, 2, 3, 4, 5, 6]:
            self.assertEqual(self.model.split_weight[0][0, i, 0], 0)
        # Test a middle split
        self.assertEqual(self.model.split_weight[2][1, 5, 1], -1)
        for i in [0, 1, 2, 3, 4, 6]:
            self.assertEqual(self.model.split_weight[2][1, i, 1], 0)
        # Test the last split
        self.assertEqual(self.model.split_weight[3][3, 3, 24], -1)
        for i in [0, 1, 2, 4, 5, 6]:
            self.assertEqual(self.model.split_weight[3][3, i, 24], 0)

    def test_split_weight_defaults(self):
        # Test a split that's never reached
        for i in range(7):
            self.assertEqual(self.model.split_weight[2][0, i, 8], 0)

    def test_split_strength_values(self):
        self.assertEqual(self.model.split_strength[0][0, 0, 0], self.model.default_split_strength)
        self.assertEqual(self.model.split_strength[2][1, 0, 1], self.model.default_split_strength)
        self.assertEqual(self.model.split_strength[3][3, 0, 24], self.model.default_split_strength)

    def test_split_strength_defauls(self):
        self.assertEqual(self.model.split_strength[2][0, 0, 8], 0)
