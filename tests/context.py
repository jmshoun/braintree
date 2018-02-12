import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import braintree.data as data
import braintree.tree as tree
import braintree.neural as neural

# Sample data sets for testing
concrete = np.loadtxt("tests/test_data/concrete.csv", delimiter=",", skiprows=10)
