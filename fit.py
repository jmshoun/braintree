import pickle

import braintree

with open("data/song_matrix.p", "rb") as infile:
    songs = pickle.load(infile)
num_predictors = songs["train"]["X"].shape[1]

train_data = braintree.TensorFlowData(songs["train"]["X"], songs["train"]["y"])
validation_data = braintree.TensorFlowData(songs["validation"]["X"], songs["validation"]["y"])
model = braintree.BrainTree(num_predictors, num_trees=5, max_depth=2,
                            batch_size=8, learning_rate=0.01)
model.initialize(songs["train"])
model.train(train_data, validation_data, training_steps=1000, print_every=100)
