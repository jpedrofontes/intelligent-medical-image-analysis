# Suppress python warnings about deprecations
import sys
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")

# Functions to load and save a classifier
import pickle

def load_model():
    pass

def save_model():
    pass

if __name__ == "__main__":
    # Argument parsing
    import os
    import argparse

    parser = argparse.ArgumentParser(description='Train a system to identify benign and malign tumors from the BCDR dataset.')
    parser.add_argument('instances', metavar='instance', type=str, nargs='+', help='BCDR instances to use in the training process')
    parser.add_argument('-s', '--save-rois', dest='save', action='store_true', help='save ROI\'s extracted from BCDR instances')
    parser.add_argument('-c', '--classifier', dest='classifier', help='model to be used for classification', default='MLP')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate model generated')
    parser.add_argument('-p', '--path', dest='PATH', help='path to store model files', default=os.path.join(os.getcwd(), '/model'))
    args = parser.parse_args()

    # Imports
    import keras
    import cv2 as cv
    import numpy as np

    import matplotlib.pyplot as plt

    from datasets import bcdr
    from sklearn.svm import SVC
    from sklearn.neural_network import MLPClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, confusion_matrix

    from keras.layers import Input
    from keras.preprocessing import image
    from keras.applications.inception_v3 import InceptionV3, preprocess_input

    # Tensorflow verbosity control
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    # BCDR data, shuffled and split between train and test sets
    images = np.array([]).reshape(-1, 299, 299, 1)
    labels = np.array([])
    # Join instance images
    for instance in args.instances:
        (img, lbs) = bcdr.load_data(instance, save_rois=args.save, target_size=(299, 299, 1))
        images = np.append(images, img, axis=0)
        labels = np.append(labels, lbs, axis=0)
    # Split in train and test data
    x_train, x_test, y_train, y_test = train_test_split(
        images,
        labels,
        test_size=0.33,
        random_state=123,
        stratify=labels)

    # Feature Extraction Model
    print('[INFO] Loading InceptionV3 model for feature extration...\n')
    model = InceptionV3(weights='imagenet', include_top=False)

    if args.classifier == 'SVM':
        # Support Vector Machines Classification
        if not os.path.isfile(os.path.join(os.getcwd(), args.PATH, '.svm')):
            # Extract features from CNN network
            print('[INFO] Extracting features...\n')
            features = []
            labels = []
            for img in x_train:
                x = np.expand_dims(img, axis=0)
                x = preprocess_input(x)
                features_temp = model.predict(x)
                features.append(features_temp.flatten())
            labels = y_train.flatten().astype(int)
            unique, counts = np.unique(labels, return_counts=True)
            # Fit SVM model to the features extracted
            print('[INFO] Training SVM model...\n')
            clf = SVC(C=0.0001, class_weight='balanced')
            clf.fit(features, labels)
            serialized = pickle.dumps(clf)
            with open(os.path.join(os.getcwd(), args.PATH, '.svm'), "wb") as f:
                f.write(serialized)
        else:
            print('[INFO] Loading pretrained SVM model...\n')
            with open(os.path.join(os.getcwd(), args.PATH, '.svm'), "rb") as f:
                serialized = f.read()
            clf = pickle.loads(serialized)
    elif args.classifier == 'MLP':
        # Multi Layer Perceptron Classification
        if not os.path.isfile(os.path.join(os.getcwd(), args.PATH, '.mlp')):
            # Extract features from CNN network
            print('[INFO] Extracting features...\n')
            features = []
            labels = []
            for img in x_train:
                x = np.expand_dims(img, axis=0)
                x = preprocess_input(x)
                features_temp = model.predict(x)
                features.append(features_temp.flatten())
            labels = y_train.flatten()
            # Fit MLP model to the features extracted
            print('[INFO] Training MLP model...\n')
            clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                hidden_layer_sizes=(512, 256, 128), random_state=123)
            clf.fit(features, labels)
            serialized = pickle.dumps(clf)
            with open(os.path.join(os.getcwd(), args.PATH, '.mlp'), "wb") as f:
                f.write(serialized)
        else:
            print('[INFO] Loading pretrained MLP model...\n')
            with open(os.path.join(os.getcwd(), args.PATH, '.mlp'), "rb") as f:
                serialized = f.read()
            clf = pickle.loads(serialized)
    else:
        print('Classifier not supported')
        sys.exit(1)
    # Model Evaluation
    real, pred = [], []
    for i in range(len(x_test)):
        x = np.expand_dims(x_test[i], axis=0)
        x = preprocess_input(x)
        features_temp = model.predict(x)
        features = features_temp.flatten()
        real.append(y_test[i])
        prediction = clf.predict([features])
        pred.append(prediction[0])
    print('[INFO] Accuracy: {}\n\n[INFO] Confusion Matrix:\n{}'.format(accuracy_score(real, pred), confusion_matrix(real, pred)))
