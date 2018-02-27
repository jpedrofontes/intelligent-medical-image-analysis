# Suppress python warnings about deprecations
import warnings
warnings.simplefilter("ignore")

# Imports
import os
import sys
import argparse
import cv2 as cv
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from datasets import BCDR as bcdr
from classifiers import SimpleMLP, SimpleCNN, AlexNet

tf.logging.set_verbosity(tf.logging.INFO)

# Argument parsing
parser = argparse.ArgumentParser(description='Train a system to identify benign and malign tumors from the BCDR dataset.')
parser.add_argument('instance', metavar='instance', type=str, nargs='+', help='BCDR instances to use in the training process')
parser.add_argument('-t', '--train', dest='train', action='store_true', help='start train from scratch')
parser.add_argument('-p', '--path', dest='PATH', help='path to store model files', default=os.path.join(os.getcwd(), '/model'))
args = parser.parse_args()

# BCDR data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = bcdr.load_data(args.instance[0])

num_classes = 2
epochs = 10

# Create estimator => high-level model training
classifier = tf.estimator.Estimator(model_fn=AlexNet.createModel, model_dir=args.PATH)
# Create logger
tensors_to_log = {  }
logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)

# Train the model
if args.train:
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x = {"x": np.array(x_train, dtype=np.float32)},
        y = np.array(y_train),
        batch_size = 128,
        num_epochs = None,
        shuffle = True)
    classifier.train(
        input_fn = train_input_fn,
        steps = 2048,
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
