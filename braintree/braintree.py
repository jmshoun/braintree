import copy

from . import neural
from . import tree


class NotFitError(Exception):
    """Exception to raise when predict is called before fit on a BrainTree object."""
    pass


class BrainTree(object):
    """Interface to a holistic BrainTree model.
    
    Attributes:
        tree (TreeModel): GBM model fit via xgboost used to initialize the model.
        neural (NeuralModel): Continuous extension of GBM model trained via TensorFlow.
    """

    def __init__(self, num_trees, standardize=True, max_depth=4, subsample=0.5, eta=0.1,
                 default_split_strength=2, train_steps=1000, learning_rate=0.01, batch_size=32,
                 dropout_rate=0.5, split_weight_noise=0.01, split_bias_noise=0.01,
                 terminal_weight_noise=0.01):
        """Default constructor.
        
        Args:
            num_trees (int): Number of decision trees in the model.
            max_depth (int): Maximum depth (number of consecutive splits) for each tree.
            subsample (float): Subsampling rate to use when training the trees, [0, 1].
            default_split_strength (float): Initial strength (or "sharpness") of each tree split
                when the neural model starts training. Multiplier on the logit scale.
            train_steps (int): Number of training steps for the neural model, >0.
            learning_rate (float): Learning rate for the neural model, (0, 1].
            batch_size (int): Batch size for the neural model, >0.
            dropout_rate (float): Dropout rate when training the neural model, (0, 1].        
        """
        self.tree = tree.TreeModel(num_trees, max_depth, subsample,
                                   default_split_strength, eta)
        # We can't initialize the neural model until we know the number of predictors.
        self.neural = None
        self.train_steps = train_steps
        self.standard_factors = {} if standardize else None
        self._neural_config = {"num_trees": num_trees, "max_depth": max_depth,
                               "learning_rate": learning_rate, "batch_size": batch_size,
                               "dropout_rate": dropout_rate}
        self._noise_config = {"split_weight_noise": split_weight_noise,
                              "split_bias_noise": split_bias_noise,
                              "terminal_weight_noise": terminal_weight_noise}

    def fit(self, train_data, validation_data, print_every=0, seed=0):
        """Fits the model to provided train and validation data sets.
        
        Args:
            train_data (BrainTreeData): Training data set for the model.
            validation_data (BrainTreeData): Validation data set for the model.
            print_every (int): Number of training steps after which to print model diagnostics.
            seed (int): Seed for the random generators at all stages of model training.
        """
        if self.standard_factors is not None:
            train_data = copy.deepcopy(train_data)
            train_data.standardize()
            self.standard_factors = train_data.standard_factors
            validation_data = copy.deepcopy(validation_data)
            validation_data.standardize(self.standard_factors)
        self.tree.fit(train_data.to_dmatrix(), validation_data.to_dmatrix(), seed=seed)
        self.neural = neural.NeuralModel(train_data.num_features, **self._neural_config)
        self.neural.load_params(self.tree, **self._noise_config)
        self.neural.train(train_data, validation_data, self.train_steps, print_every)

    def predict(self, data):
        """Predicts outcomes for the given data using the model.
        
        Args:
            data (BrainTreeData): Data to score through the model.
        Returns:
            np.ndarray: The predicted response for each observation in the provided data set.
        """
        if self.neural is None:
            raise NotFitError("The model has not been fit yet!")
        if self.standard_factors and not data.standardized:
            data = copy.deepcopy(data)
            data.standardize(self.standard_factors)
        predictions = self.neural.predict(data)
        if self.standard_factors:
            predictions = (predictions * self.standard_factors["response_sds"][0]
                           + self.standard_factors["response_means"][0])
        return predictions
