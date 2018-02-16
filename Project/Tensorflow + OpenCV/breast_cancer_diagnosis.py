# Suppress python warnings about deprecations
import warnings
warnings.simplefilter("ignore")

# Imports
import os
import sys
import cv2 as cv
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from datasets import BCDR as bcdr
from classifiers import SimpleMLP, AlexNet, CNN

tf.logging.set_verbosity(tf.logging.INFO)

num_classes = 2
epochs = 10

# BCDR data, shuffled and split between train and test sets
if len(sys.argv) == 1:
    (x_train, y_train), (x_test, y_test) = bcdr.load_data()
else:
    (x_train, y_train), (x_test, y_test) = bcdr.load_data(sys.argv[1])

# Create estimator => high-level model training
classifier = tf.estimator.Estimator(model_fn=CNN.createModel, model_dir="/tmp/convnet_model")
# Create logger
tensors_to_log = { "probabilities": "softmax_tensor" }
logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)

# Train the model
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x = {"x": np.array(x_train, dtype=np.float32)},
    y = np.array(y_train),
    batch_size = 100,
    num_epochs = None,
    shuffle = True)
classifier.train(
    input_fn = train_input_fn,
    steps = 1000,
    hooks = [logging_hook])

# Evaluate the model and print results
eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    x = {"x": np.array(x_test, dtype=np.float32)},
    y = np.array(y_test),
    num_epochs = 1,
    shuffle = False)
eval_results = classifier.evaluate(
    input_fn=eval_input_fn)
print(eval_results)


# Create network
# net = SimpleMLP([None, 32, 32], num_classes=num_classes)
# net = AlexNet([None, 32, 32], num_classes=num_classes)
# net_model = net.createModel()

# Train network
# with tf.Session() as sess:
#     tf.set_random_seed(1234)
#     sess.run(tf.global_variables_initializer())
#     # Compute epochs
#     for i in range(epochs):
#         # Train network
#         _, loss_value = sess.run([net.train_op, net.loss], feed_dict={net.x: x_train, net.y: y_train})
#         # Calculate correct matches
#         predicted = sess.run([net.correct_pred], feed_dict={net.x: x_test})[0]
#         match_count = sum([int(y == y_) for y, y_ in zip(y_test, predicted)])
#         # Calculate the accuracy
#         accuracy = match_count / len(y_test)
#         # Print the epoch results
#         print('Finished Epoch[{} of {}]: [Training] loss = {:.3f}, accuracy = {:.3f};'.format(i+1, epochs, accuracy, loss_value))
