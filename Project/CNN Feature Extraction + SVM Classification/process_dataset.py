# Suppress python warnings about deprecations
import sys
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")

# Argument parsing
import os
import argparse

parser = argparse.ArgumentParser(description='Train a system to identify benign and malign tumors from the BCDR dataset.')
parser.add_argument('instances', metavar='instance', type=str, nargs='+', help='BCDR instances to use in the training process')
parser.add_argument('-t', '--train', dest='train', action='store_true', help='start train from scratch')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate model generated')
parser.add_argument('-p', '--path', dest='PATH', help='path to store model files', default=os.path.join(os.getcwd(), '/model'))
args = parser.parse_args()

# Imports
import keras
import pickle
import cv2 as cv
import numpy as np

import matplotlib.pyplot as plt

from skimage import io
from sklearn import svm
from datasets import BCDR as bcdr

from keras.layers import Input
from keras.preprocessing import image
from keras.applications.inception_v3 import InceptionV3, preprocess_input

# Tensorflow verbosity control
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# BCDR data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = bcdr.load_data(args.instances[0])

if not os.path.isdir(os.path.join(os.getcwd(), 'BCDR')):
    print('[INFO] Saving ROI\'s extracted...\n', )
    # kernel = np.ones((5,5),np.uint8)
    os.mkdir('BCDR')
    os.mkdir('BCDR/benign')
    os.mkdir('BCDR/malign')
    for i in range(x_train.shape[0]):
        if y_train[i] == 0:
            io.imsave(os.path.join('BCDR/benign', '{}.jpg'.format(i)), x_train[i])
        else:
            io.imsave(os.path.join('BCDR/malign', '{}.jpg'.format(i)), x_train[i])
        # x_train[i] = cv.morphologyEx(x_train[i], cv.MORPH_OPEN, kernel)
        # x_train[i] = cv.equalizeHist(x_train[i])

# Feature Extraction Model
model = InceptionV3(weights='imagenet', include_top=False)

# Support Vector Machines Classification
if not os.path.isfile(os.path.join(os.getcwd(), args.PATH)):
    benign_path = os.path.join(os.getcwd(), 'BCDR/benign')
    malign_path = os.path.join(os.getcwd(), 'BCDR/malign')
    features = []
    labels = []

    for f in os.listdir(benign_path):
        if os.path.isfile(os.path.join(benign_path, f)):
            img_path = os.path.join(benign_path, f)
            img = image.load_img(img_path, target_size=(299, 299))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            features_temp = model.predict(x)
            features.append(features_temp.flatten())
            labels.append(0)

    for f in os.listdir(malign_path):
        if os.path.isfile(os.path.join(malign_path, f)):
            img_path = os.path.join(malign_path, f)
            img = image.load_img(img_path, target_size=(299, 299))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            features_temp = model.predict(x)
            features.append(features_temp.flatten())
            labels.append(1)
            
    clf = svm.SVC()
    clf.fit(features, labels)
    serialized = pickle.dumps(clf)
    with open(os.path.join(os.getcwd(), args.PATH), "wb") as f:
        f.write(serialized)
else:
    with open(os.path.join(os.getcwd(), args.PATH), "rb") as f:
        serialized = f.read()
    clf = pickle.loads(serialized)

img_path = 'BCDR/benign/2.jpg'
img = image.load_img(img_path, target_size=(299, 299))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
features_x = model.predict(x)
pred = clf.predict([features_x.flatten()])
print(pred)

img_path = 'BCDR/malign/0.jpg'
img = image.load_img(img_path, target_size=(299, 299))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
features_x = model.predict(x)
pred = clf.predict([features_x.flatten()])
print(pred)
