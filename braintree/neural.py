"""Primary BrainTree model class."""

import tensorflow as tf
import numpy as np

_GRADIENT_CAP_PERCENTILE = 90


def _zeros(*shape):
    """Creates a TensorFlow Variable of the given shape filled with zeros."""
    return tf.Variable(np.zeros(shape), dtype="float32")


class NeuralModel(object):
    """Neural network component of a BrainTree model."""
    def __init__(self, num_features, num_trees=50, max_depth=4, learning_rate=0.01,
                 batch_size=32, dropout_rate=0.5):
        # Basic structure
        self.num_features = num_features
        self.num_trees = num_trees
        self.max_depth = max_depth
        # Tuning parameters
        self.batch_size = batch_size
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        # Components of the graph
        self.predictors = self.response = self.dropout = None
        self.split_weight = self.split_bias = self.split_strength = None
        self.terminal_weight = self.terminal_bias = None
        self.predictions = self.loss = self.optimizer = None
        self.fitted_gradient_limits = None
        # Graph building
        self.graph = tf.Graph()
        with self.graph.as_default():  # pylint: disable=not-context-manager
            self._build_graph()
            self.session = tf.Session(graph=self.graph)
            self.saver = tf.train.Saver()
            self.session.run(tf.global_variables_initializer())

    @property
    def num_gradients(self):
        return 3 * self.max_depth + 2

    @property
    def default_gradient_limits(self):
        max_float = np.finfo(np.float32).max
        return np.array([max_float] * self.num_gradients)

    def load_params(self, tree, split_weight_noise=0.0, split_bias_noise=0.0,
                    terminal_weight_noise=0.0):
        terminal_weight_sd = terminal_weight_noise / np.sqrt(self.num_features)
        tree.add_noise(split_weight_noise, split_bias_noise)
        with self.session.as_default():
            self.terminal_weight.load(np.random.normal(scale=terminal_weight_sd,
                                                       size=self.terminal_weight.shape))
            self.terminal_bias.load(tree.terminal_bias)
            for depth in range(self.max_depth):
                self.split_bias[depth].load(tree.split_bias[depth])
                self.split_strength[depth].load(tree.split_strength[depth])
                self.split_weight[depth].load(tree.split_weight[depth])

    def train(self, train_data, validation_data, train_steps, print_every=0):
        self._calibrate(train_data)
        current_step = 0
        train_generator = train_data.to_array_generator(self.batch_size, repeat=True)
        num_steps = print_every if print_every > 0 else train_steps
        while current_step < train_steps:
            self._train_steps(train_generator, num_steps)
            current_step += num_steps
            if print_every > 0:
                print("{:>7} - {:0.4f}".format(current_step, self.score(validation_data)))

    def _calibrate(self, train_data, num_steps=100):
        train_generator = train_data.to_array_generator(self.batch_size, repeat=True)
        gradients = []
        for _ in range(num_steps):
            predictors, responses = next(train_generator)
            input_dict = {"predictors:0": predictors,
                          "response:0": responses,
                          "dropout:0": self.dropout_rate,
                          "gradient_limits:0": self.default_gradient_limits}
            _, gradient = self.session.run([self.train_op, self.gradient_norms],
                                           feed_dict=input_dict)
            gradients += [gradient]
        gradient_array = np.array(gradients)
        gradient_caps = np.percentile(gradient_array, _GRADIENT_CAP_PERCENTILE, axis=0)
        self.fitted_gradient_limits = gradient_caps

    def _train_steps(self, train_generator, num_steps):
        for _ in range(num_steps):
            predictors, responses = next(train_generator)
            input_dict = {"predictors:0": predictors,
                          "response:0": responses,
                          "dropout:0": self.dropout_rate,
                          "gradient_limits:0": self.fitted_gradient_limits}
            _ = self.session.run([self.train_op], feed_dict=input_dict)

    def score(self, data):
        scores = []
        for (predictors, responses) in data.to_array_generator(self.batch_size):
            input_dict = {"predictors:0": predictors,
                          "response:0": responses,
                          "dropout:0": 1.0}
            loss = self.session.run([self.loss], feed_dict=input_dict)
            scores.append(loss)
        return np.mean(scores)

    def predict(self, data):
        predictions = []
        for predictors, _ in data.to_array_generator(self.batch_size):
            input_dict = {"predictors:0": predictors, "dropout:0": 1.0}
            (batch_predictions, ) = self.session.run([self.predictions], feed_dict=input_dict)
            predictions.append(batch_predictions)
        full_predictions = np.concatenate(predictions)
        return full_predictions[:data.num_observations]

    def _build_graph(self):
        self.predictors = tf.placeholder(tf.float32, shape=[1, self.batch_size, self.num_features],
                                         name="predictors")
        self.response = tf.placeholder(tf.float32, shape=[self.batch_size], name="response")
        self.dropout = tf.placeholder(tf.float32, None, name="dropout")
        self._build_model_parameters()
        self._build_predictions()
        self._build_optimizer()

    def _build_model_parameters(self):
        self.split_weight = [_zeros(2 ** i, self.num_features, self.num_trees)
                             for i in range(self.max_depth)]
        self.split_bias = [_zeros(2 ** i, 1, self.num_trees) for i in range(self.max_depth)]
        self.split_strength = [_zeros(2 ** i, 1, self.num_trees) for i in range(self.max_depth)]
        self.terminal_weight = _zeros(2 ** self.max_depth, self.num_features, self.num_trees)
        self.terminal_bias = _zeros(2 ** self.max_depth, 1, self.num_trees)

    def _build_predictions(self):
        split_prob = tf.stack([self._build_split_prob_layer(depth)
                               for depth in range(self.max_depth)], axis=3)
        terminal_probs = tf.reduce_prod(split_prob, axis=3)
        wide_predictors = tf.gather(self.predictors, [0] * (2 ** self.max_depth))
        terminal_preds = tf.matmul(wide_predictors, self.terminal_weight) + self.terminal_bias
        tree_preds = tf.reduce_sum(terminal_probs * terminal_preds, axis=0)
        self.predictions = tf.reduce_sum(tree_preds, axis=1)

    def _build_optimizer(self):
        self.loss = tf.losses.mean_squared_error(self.predictions, self.response)
        optimizer = tf.train.AdagradOptimizer(self.learning_rate)
        raw_gradients = optimizer.compute_gradients(self.loss)
        self.gradient_norms = [tf.norm(grad) for grad, _ in raw_gradients]
        self.gradient_limits = tf.placeholder(tf.float32, shape=[self.num_gradients],
                                              name="gradient_limits")
        max_limit_ratio = tf.reduce_max(self.gradient_norms / self.gradient_limits)
        clip_ratio = tf.maximum(max_limit_ratio, 1.0)
        clipped_gradients = [(grad / clip_ratio, var) for grad, var in raw_gradients]
        self.train_op = optimizer.apply_gradients(clipped_gradients)

    def _build_split_prob_layer(self, depth):
        affine_logit = tf.matmul(tf.gather(self.predictors, [0] * (2 ** depth)),
                                 self.split_weight[depth])
        raw_logit = affine_logit + tf.tile(self.split_bias[depth], [1, self.batch_size, 1])
        prob = tf.sigmoid(raw_logit * tf.exp(self.split_strength[depth]))
        prob_with_complement = tf.concat([prob, 1 - prob], axis=0)
        return tf.gather(prob_with_complement, self._prob_indices(depth))

    def _prob_indices(self, depth):
        depth_from_max = self.max_depth - depth
        repetitions = 2 ** (depth_from_max - 1)
        index_pattern = [i // 2 + (i % 2) * (2 ** depth) for i in range(2 ** (depth + 1))]
        return np.repeat(index_pattern, repetitions)
