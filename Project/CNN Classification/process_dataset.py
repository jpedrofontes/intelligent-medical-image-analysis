# Suppress python warnings about deprecations
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)

# Argument parsing
import os
import argparse

parser = argparse.ArgumentParser(description='Train a system to identify benign and malign tumors from the BCDR dataset.')
parser.add_argument('instance', metavar='instance', type=str, nargs='+', help='BCDR instances to use in the training process')
parser.add_argument('-t', '--train', dest='train', action='store_true', help='start train from scratch')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate model generated')
parser.add_argument('-c', '--compute', dest='compute', action='store_true', help='prepare dataset to premade model from Tensorflow examples')
parser.add_argument('-p', '--path', dest='PATH', help='path to store model files', default=os.path.join(os.getcwd(), '/model'))
args = parser.parse_args()

# Imports
import sys
import skimage
import cv2 as cv
import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt

from skimage import io
from datasets import BCDR as bcdr
from classifiers import SimpleMLP, SimpleCNN, AlexNet

# Tensorflow verbosity control
tf.logging.set_verbosity(tf.logging.INFO)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# BCDR data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = bcdr.load_data(args.instance[0])

if args.compute:
    kernel = np.ones((5,5),np.uint8)
    os.mkdir('BCDR')
    os.mkdir('BCDR/benign')
    os.mkdir('BCDR/malign')
    for i in range(x_train.shape[0]):
        if y_train[i] == 0:
            io.imsave(os.path.join('BCDR/benign', '{}.jpg'.format(i)), x_train[i])
        else:
            io.imsave(os.path.join('BCDR/malign', '{}.jpg'.format(i)), x_train[i])
        # x_train[i] = cv.morphologyEx(x_train[i], cv.MORPH_OPEN, kernel)
        x_train[i] = cv.equalizeHist(x_train[i])
        cv.imshow("cenas", x_train[i])
        cv.waitKey(0)

num_classes = 2
epochs = 10

# Create estimator => high-level model training
classifier = tf.estimator.Estimator(model_fn=AlexNet.createModel, model_dir=args.PATH)
# Create logger
tensors_to_log = { "softmax_tensor" }
logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=1)

# Train the model
if args.train:
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x = {"x": np.array(x_train, dtype=np.float32)},
        y = np.array(y_train),
        batch_size = 256,
        num_epochs = None,
        shuffle = True)
    classifier.train(
        input_fn = train_input_fn,
        steps = 1024,
        hooks = [logging_hook])

# Evaluate the model and print results
if args.evaluate:
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x = {"x": np.array(x_test, dtype=np.float32)},
        y = np.array(y_test),
        num_epochs = 1,
        shuffle = False)
    eval_results = classifier.evaluate(
        input_fn=eval_input_fn)
    print("Result: {}".format(eval_results))
