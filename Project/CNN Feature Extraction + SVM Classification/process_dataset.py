# Suppress python warnings about deprecations
import sys
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")

import pickle

from keras.models import model_from_json
from keras.models import model_from_yaml

def print_model(model,fich):
	from keras.utils import plot_model
	plot_model(model, to_file=fich, show_shapes=True, show_layer_names=True)

def print_history_accuracy(history):
	print(history.history.keys())
	plt.plot(history.history['acc'])
	plt.plot(history.history['val_acc'])
	plt.title('model accuracy')
	plt.ylabel('accuracy')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	plt.show()

def print_history_loss(history):
	print(history.history.keys())
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('model loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	plt.show()

def save_model_json(model, fich):
	model_json = model.to_json()
	with open(fich, "w") as json_file:
		json_file.write(model_json)

def save_model_yaml(model, fich):
	model_yaml = model.to_yaml()
	with open(fich, "w") as yaml_file:
		yaml_file.write(model_yaml)

def save_weights_hdf5(model, fich):
	model.save_weights(fich)

def load_model_json(fich):
	json_file = open(fich, 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	return loaded_model

def load_model_yaml(fich):
	yaml_file = open(fich, 'r')
	loaded_model_yaml = yaml_file.read()
	yaml_file.close()
	return model_from_yaml(loaded_model_yaml)

def load_weights_hdf5(model, fich):
	model.load_weights(fich)

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

    from keras.models import Model
    from keras.optimizers import Adam, SGD
    from keras.layers import Input, Dense, GlobalAveragePooling2D
    from keras.preprocessing import image
    from keras.applications.vgg16 import VGG16, preprocess_input

    # Tensorflow verbosity control
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    # BCDR data, shuffled and split between train and test sets
    images = np.array([]).reshape(-1, 224, 224, 3)
    labels = np.array([])
    # Join instance images
    for instance in args.instances:
        (img, lbs) = bcdr.load_data(instance, save_rois=args.save, target_size=(224, 224, 3))
        images = np.append(images, img, axis=0)
        labels = np.append(labels, lbs, axis=0)
    # Split in train and test data
    x_train, x_test, y_train, y_test = train_test_split(
        images,
        labels,
        test_size=0.4,
        random_state=12345,
        stratify=labels)

    # Check if a network exists
    if not os.path.isfile(os.path.join(os.getcwd(), args.PATH, '.cnn')):
        # Feature Extraction Model
        print('[INFO] Fine-tuning VGG16 model for feature extration...\n')
        base_model = VGG16(weights='imagenet', include_top=False)
        # Add classification layers
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu', name="dense_one")(x)
        predictions = Dense(2, activation='softmax', name="dense_two")(x)
        # this is the model we will train
        model = Model(inputs=base_model.input, outputs=predictions)
        # first: train only the top layers (which were randomly initialized)
        # i.e. freeze all convolutional VGG16 layers
        # for layer in base_model.layers:
        #     layer.trainable = False
        # compile the model (should be done *after* setting layers to non-trainable)
        model.compile(optimizer=SGD(lr=0.001, decay=1e-9, momentum=0.9, nesterov=True),
            loss='categorical_crossentropy',
            metrics=['accuracy'])
        # prepare a tensorboard callback
        tbCallBack = keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)
        # train the model on the new data for a few epochs
        from keras.utils import to_categorical
        model.fit(x_train, to_categorical(y_train, num_classes=2), batch_size=1, epochs=150, verbose=2, callbacks=[tbCallBack], validation_split=0.1, shuffle=True)
        # at this point, the top layers are well trained and we can start fine-tuning
        # convolutional layers from inception V3. We will freeze the bottom N layers
        # and train the remaining top layers.
        # let's visualize layer names and layer indices to see how many layers
        # we should freeze:
        # for i, layer in enumerate(base_model.layers):
        #    print(i, layer.name)
        # we chose to train the top 2 inception blocks, i.e. we will freeze
        # the first 21 layers and unfreeze the rest:
        # for layer in model.layers[:21]:
        #    layer.trainable = False
        # for layer in model.layers[21:]:
        #    layer.trainable = True
        # we need to recompile the model for these modifications to take effect
        # we use SGD with a low learning rate
        # model.compile(optimizer=SGD(lr=0.001, decay=1e-6, momentum=0.5, nesterov=True),
        #     loss='categorical_crossentropy',
        #     metrics=['accuracy','mse'])
        # we train our model again (this time fine-tuning the top 2 inception blocks
        # alongside the top Dense layers
        # model.fit(x_train, keras.utils.to_categorical(y_train, num_classes=2), batch_size=2, epochs=150, verbose=2, callbacks=[tbCallBack], validation_split=0.2, shuffle=True)
        # Create model with base model outputs for feature extraction
        model = Model(inputs=base_model.input, outputs=model.get_layer("dense_one").output)
        # Save model
        save_model_json(model=model, fich='./.cnn')
        save_weights_hdf5(model=model, fich='./.cnn_weights')
    else:
        model = load_model_json(fich='./.cnn')
        load_weights_hdf5(model=model, fich='./.cnn_weights')
    # choose classifier
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
