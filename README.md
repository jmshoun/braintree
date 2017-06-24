# braintree

### Neural Network Representations of Decision Trees

Deep neural networks are great for certain domains like image processing and natural language processing. However, tree-based ensembles like GBMs (via xgboost) tend to dominate in structured regression applications. GBMs are able to achieve remarkable performance in spite of the lack of expressive power of indvidual models in the ensembles, whereas neural networks are far more expressive. However, what neural networks have in expressive power, they lack in ease of training.

What if we try building a neural network structure that directly maps to a superset of GBMs? We can relax the constraints of decision trees as follows:
* Instead of forcing each terminal node to be a constant, let it be a linear function of all of the predictors.
* Instead of forcing each decision boundary to be a function of one predictor, let it be a linear function of all of the predictors.
* Instead of forcing the choice between terminal nodes to be a hard threshold, let it be a sigmoid function and take the weighted mean of the set of all terminal node predictions.

Finally, instead of training this network from scratch, why not seed it with the parameters from a GBM? In theory, the performance of the neural network would then be no worse than the GBM.

Will this method provide any significant performance improvement? Will it be unworkably slow? Let's find out!
