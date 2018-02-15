# Suppress python warnings about deprecations
import warnings
warnings.simplefilter("ignore")

# Imports
import os
import cv2 as cv
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from datasets import BCDR as bcdr
from classifiers import SimpleMLP, AlexNet

num_classes = 2
epochs = 200

# BCDR data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = bcdr.load_data(bcdr.F01)

# Create network
net = SimpleMLP([None, 32, 32], num_classes=num_classes)
net_model = net.createModel()

# Train network
with tf.Session() as sess:
    tf.set_random_seed(1234)
    sess.run(tf.global_variables_initializer())
    # Compute epochs
    for i in range(epochs):
        # Train network
        _, loss_value = sess.run([net.train_op, net.loss], feed_dict={net.x: x_train, net.y: y_train})
        # Calculate correct matches
        predicted = sess.run([net.correct_pred], feed_dict={net.x: x_test})[0]
        match_count = sum([int(y == y_) for y, y_ in zip(y_test, predicted)])
        # Calculate the accuracy
        accuracy = match_count / len(y_test)
        # Print the epoch results
        print('Epoch {} done :: Accuracy: {:.3f} :: Loss: {:.3f}'.format(i+1, accuracy, loss_value))
