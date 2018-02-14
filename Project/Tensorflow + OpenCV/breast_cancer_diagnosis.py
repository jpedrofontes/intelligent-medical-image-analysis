# Suppress python warnings about deprecations
import warnings
warnings.simplefilter("ignore")

# Imports
import os
import bcdr
import cv2 as cv
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from classifiers import AlexNet as net

# BCDR data, shuffled and split between train and test sets:
(x_train, y_train), (x_test, y_test) = bcdr.load_data('BCDR-F01')
