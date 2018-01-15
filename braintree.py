import datetime
import math
import re
import os

import numpy as np
import tensorflow as tf
import xgboost as xgb


class TensorFlowModel(object):
    def __init__(self):
        self.graph = tf.Graph()
        self.fit_log = {"batch_number": [],
                        "validation_score": []}
        self.fit_time = None

    def train(self, train_data, validation_data, training_steps=1000, print_every=200):
        """Train the model.
        Inputs:
            train_data: TensorFloWData object with training data.
            validation_date: TensorFlowData object with validation data.
            training_steps: Number of batches to train.
            print_every: How often to calculate and show validation results.
        """

        start_time = datetime.datetime.now()
        current_step = 0
        while current_step < training_steps:
            self._train_steps(train_data, print_every)
            current_step += print_every
            current_score = self.score(validation_data)
            print("{:>7} - {:0.4f}".format(current_step, current_score))
            self.fit_log["batch_number"].append(current_step)
            self.fit_log["validation_score"].append(current_score)

        end_time = datetime.datetime.now()
        self.fit_time = (end_time - start_time).total_seconds()

    def _train_steps(self, train_data, num_steps):
        for _ in range(num_steps):
            input_dict = train_data.get_batch(self.batch_size)
            input_dict["dropout:0"] = self.dropout_rate
            _, loss = self.session.run([self.optimizer, self.loss], feed_dict=input_dict)

    def score(self, data):
        """Score the model.
        Inputs:
            data: A TensorFlowData object with data to score the model on.
        """
        scores = []
        while not data.has_reached_end():
            input_dict = data.get_batch(self.batch_size)
            input_dict["dropout:0"] = 1.0
            loss = self.session.run([self.loss], feed_dict=input_dict)
            scores.append(loss)
        return math.sqrt(np.mean(scores))

    def save(self, filename):
        """Save the values of all model variables to a file."""
        # TODO: Create folder if it doesn't exist
        self.saver.save(self.session, filename)

    def restore(self, filename):
        """Restore the values of all model variables from a file."""
        self.saver.restore(self.session, filename)

    @staticmethod
    def random_variable(shape, stddev=0.01):
        return tf.Variable(tf.random_normal(shape, stddev=stddev))


class BrainTree(TensorFlowModel):
    def __init__(self, num_features, num_trees, max_depth,
                 batch_size=32, learning_rate=0.001, dropout_rate=0.5,
                 split_weight_stddev=0.01, terminal_weight_stddev=0.01,
                 eta=0.5, subsample=0.5, initial_strength=2):
        super().__init__()
        self.num_features = num_features
        self.num_trees = num_trees
        self.max_depth = max_depth
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.split_weight_stddev = split_weight_stddev
        self.terminal_weight_stddev = terminal_weight_stddev
        self.initial_strength = initial_strength
        # Xgboost initialization parameters
        self.eta = eta
        self.subsample = subsample

        with self.graph.as_default():
            self._build_graph()
            self.session = tf.Session(graph=self.graph)
            self.saver = tf.train.Saver()
        self.node_names = [node.name + ":0" for node in self.graph.as_graph_def().node]

    def initialize(self, train_data):
        with self.graph.as_default():
            self.session.run(tf.global_variables_initializer())

        gbm_model = self._fit_gbm(train_data)
        gbm_parameters = self._gbm_to_params(gbm_model)
        with self.session.as_default():
            self.terminal_bias.load(gbm_parameters["terminal_bias"])
            for i in range(self.max_depth):
                self.split_bias[i].load(gbm_parameters["split_bias"][i])
                self.split_strength[i].load(gbm_parameters["split_strength"][i])
                self.split_weight[i].load(gbm_parameters["split_weight"][i])

    def _fit_gbm(self, train_data):
        """Fits a GBM to training data."""
        xgb_data = xgb.DMatrix(train_data["X"], label=train_data["y"])
        return xgb.train({"eta": self.eta, "max_depth": self.max_depth,
                          "subsample": self.subsample},
                         dtrain=xgb_data, num_boost_round=self.num_trees,
                         evals=[(xgb_data, "train")],
                         verbose_eval=True)

    def _gbm_to_params(self, model):
        tree_filename = ".tree_dump.tmp"
        model.dump_model(tree_filename)
        with open(tree_filename, "r") as tree_file:
            tree_lines = tree_file.read().split("\n")[:-1]
        trees = self._split_trees(tree_lines)
        os.remove(tree_filename)

        params = {"terminal_bias": np.zeros([2 ** self.max_depth, 1, self.num_trees]),
                  "split_bias": [np.zeros([2 ** depth, 1, self.num_trees])
                                 for depth in range(self.max_depth)],
                  "split_strength": [np.zeros([2 ** depth, 1, self.num_trees]) for depth in
                                     range(self.max_depth)],
                  "split_weight": [np.random.randn(2 ** depth, self.num_features, self.num_trees)
                                   * self.split_weight_stddev / math.sqrt(self.num_features)
                                   for depth in range(self.max_depth)]}

        for i, tree in enumerate(trees):
            params = self._parse_tree(tree, i, params)
        return params

    @staticmethod
    def _split_trees(tree_lines):
        trees = []
        current_tree = []
        for line in tree_lines[1:]:
            if line.find("booster") == 0:
                trees.append(current_tree)
                current_tree = []
            else:
                current_tree.append(line)
        trees.append(current_tree)
        return trees

    def _parse_tree(self, tree, tree_number, params):
        leading_tabs_re = re.compile(r"[^\t]")
        split_re = re.compile(r"f(?P<predictor>[0-9]+)<(?P<bias>-?[0-9]+(\.[0-9]+)?)")
        terminal_re = re.compile(r"leaf=(?P<bias>-?[0-9]+(\.[0-9]+)?)")
        current_index = 0
        for line in tree:
            depth = leading_tabs_re.search(line).start()
            split_match = split_re.search(line)
            if split_match:
                split_predictor = int(split_match.group("predictor"))
                split_bias = float(split_match.group("bias"))
                split_index = current_index // (2 ** (self.max_depth - depth))
                params["split_bias"][depth][split_index, 0, tree_number] = split_bias
                params["split_weight"][depth][split_index, split_predictor, tree_number] = -1
                params["split_strength"][depth][split_index, 0, tree_number] = self.initial_strength
            else:
                terminal_match = terminal_re.search(line)
                terminal_bias = float(terminal_match.group("bias"))
                for _ in range(2 ** (self.max_depth - depth)):
                    params["terminal_bias"][current_index, 0, tree_number] = terminal_bias
                    current_index += 1

        return params

    def _build_graph(self):
        self.predictors, self.response = self._build_inputs()
        self.dropout = tf.placeholder(tf.float32, None, name="dropout")

        # Model parameters
        self.split_weight = [self.random_variable([2 ** i, self.num_features, self.num_trees])
                             for i in range(self.max_depth)]
        self.split_bias = [self.random_variable([2 ** i, 1, self.num_trees])
                           for i in range(self.max_depth)]
        self.split_strength = [self.random_variable([2 ** i, 1, self.num_trees])
                               for i in range(self.max_depth)]
        self.terminal_weight = self.random_variable([2 ** self.max_depth, self.num_features,
                                                     self.num_trees],
                                                    stddev=self.terminal_weight_stddev
                                                           / math.sqrt(self.num_features))
        self.terminal_bias = self.random_variable([2 ** self.max_depth, 1, self.num_trees])

        # Optimization
        self.pred = self._build_predictions()
        self.loss = tf.losses.mean_squared_error(self.pred, self.response)
        self.optimizer = tf.train.AdagradOptimizer(self.learning_rate).minimize(self.loss)

    def _build_predictions(self):
        self.split_prob = tf.stack([self._build_split_prob(depth)
                                   for depth in range(self.max_depth)], axis=3)
        self.terminal_prob = tf.reduce_prod(self.split_prob, axis=3)
        self.terminal_pred = tf.matmul(tf.gather(self.predictors, [0] * (2 ** self.max_depth)),
                                       self.terminal_weight) + self.terminal_bias
        self.tree_pred = tf.reduce_sum(self.terminal_prob * self.terminal_pred, axis=0)
        return tf.reduce_sum(self.tree_pred, axis=1)

    def _build_split_prob(self, depth):
        basic_logit = tf.matmul(tf.gather(self.predictors, [0] * (2 ** depth)),
                                self.split_weight[depth]) \
                      + tf.tile(self.split_bias[depth], [1, self.batch_size, 1])
        prob = tf.sigmoid(basic_logit * tf.exp(self.split_strength[depth]))
        prob_with_complement = tf.concat([prob, 1 - prob], axis=0)
        return tf.gather(prob_with_complement,
                         np.repeat([i // 2 + (i % 2) * (2 ** depth)
                                    for i in range(2 ** (depth + 1))],
                                   (2 ** (self.max_depth - depth - 1))))

    def _build_inputs(self):
        predictors = tf.placeholder(tf.float32, shape=[1, self.batch_size, self.num_features],
                                    name="predictors")
        response = tf.placeholder(tf.float32, shape=[self.batch_size], name="response")
        return predictors, response
