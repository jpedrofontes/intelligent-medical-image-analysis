# Suppress python warnings about deprecations
import sys
import warnings
if not sys.warnoptions:
	warnings.simplefilter("ignore")

import gc
import pickle

from keras.models import model_from_json
from keras.models import model_from_yaml

def print_model(model, fich):
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
	parser.add_argument('-c', '--classifier', dest='classifier', help='model to be used for classification', default=None)
	parser.add_argument('-p', '--path', dest='PATH', help='path to store model files', default=os.path.join(os.getcwd(), '/model'))
	parser.add_argument('-g', '--gpus', dest='gpus', help='number of GPU\'s to use for training', default=None)
	parser.add_argument('-b', '--batch-size', dest='batch_size', help='number of batches to use for training', default=8)
	parser.add_argument('-v', '--cross-validation', dest='validation', help='whether to evaluate model using cross validation (10-fold)', action='store_true')
	parser.add_argument('-e', '--epochs', dest='epochs', help='number of training epochs', default=30)
	parser.add_argument('-d', '--device', dest='device', help='where to run the training (GPU index)', default=0)
	args = parser.parse_args()

	# Imports
	import time
	import keras
	import cv2 as cv
	import numpy as np
	import tensorflow as tf
	import matplotlib.pyplot as plt

	from datasets import bcdr
	from sklearn.svm import SVC
	from sklearn.decomposition import PCA
	from sklearn.model_selection import StratifiedKFold, train_test_split
	from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc

	from keras import backend as K
	from keras.layers import *
	from keras.models import Model
	from keras.optimizers import Adam, SGD
	from keras.preprocessing import image
	from keras.utils import to_categorical
	from keras.utils.multi_gpu_utils import multi_gpu_model
	from keras.applications.inception_v3 import InceptionV3, preprocess_input

	# Tensorflow verbosity control
	os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

	import tensorflow as tf

	with tf.device('/device:GPU:{}'.format(args.device)):
		# Numpy seed, for reproducibility
		np.random.seed(123)

		# BCDR data, shuffled and split between train and test sets
		images = np.array([]).reshape(-1, 224, 224, 3)
		labels = np.array([])
		# Join instance images
		for instance in args.instances:
			(img, lbs) = bcdr.load_data(instance, save_rois=args.save, target_size=(224, 224, 3))
			images = np.append(images, img, axis=0)
			labels = np.append(labels, lbs, axis=0)
		# Split in train and test data
		kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=12345)
		cvscores = []
		i = 1
		# Check if a network exists
		if not os.path.isfile(os.path.join(os.getcwd(), args.PATH, 'cnn_model.json')):
			# Feature Extraction Model
			print('[INFO] Fine-tuning InceptionV3 model for feature extration...\n')
			# prepare a tensorboard callback
			tbCallBack = keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)
			if args.validation:
				# train the model using 10-Fold Cross Validation
				for train, test in kfold.split(images, labels):
					print('[TRAINING] Iteration {}:'.format(i))
					# Measure time
					start = time.time()
					# Load base model
					base_model = InceptionV3(weights='imagenet', include_top=False)
					# Add classification layers
					x = base_model.output
					x = GlobalAveragePooling2D()(x)
					x = Dense(1024, activation='relu', name="dense_1")(x)
					x = Dense(512, activation='relu', name="dense_2")(x)
					predictions = Dense(2, activation='softmax', name="dense_3")(x)
					# this is the model we will train
					model = Model(inputs=base_model.input, outputs=predictions)
					if args.gpus is not None:
						model = multi_gpu_model(model, gpus=int(args.gpus))
					# Compile model
					model.compile(optimizer=SGD(lr=0.001, decay=1e-9, momentum=0.9, nesterov=True),
						loss='categorical_crossentropy',
						metrics=['accuracy'])
					# Fit the model
					model.fit(images[train], to_categorical(labels[train]), batch_size=int(args.batch_size), epochs=args.epochs, verbose=1, callbacks=[tbCallBack], shuffle=True, validation_data=(images[test], to_categorical(labels[test])))
					# Measure and print execution time
					end = time.time()
					print('[METRICS] Execution time: {}ms'.format(end-start))
					# Evaluate the model
					scores = model.evaluate(images[test], to_categorical(labels[test]), verbose=0)
					print('[RESULTS] Accuracy: {}%'.format(scores[1]*100))
					cvscores.append(scores[1] * 100)
					gc.collect()
					i = i+1
				print('[RESULTS] Final Results: {}% (+/- {}%)'.format(np.mean(cvscores), np.std(cvscores)))
			else:
				# Split train and test data
				x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=12345)
				# Measure time
				start = time.time()
				# Load base model
				base_model = InceptionV3(weights='imagenet', include_top=False)
				# Add classification layers
				x = base_model.output
				x = GlobalAveragePooling2D()(x)
				x = Dense(1024, activation='relu', name="dense_1")(x)
				x = Dense(512, activation='relu', name="dense_2")(x)
				predictions = Dense(2, activation='softmax', name="dense_3")(x)
				# this is the model we will train
				model = Model(inputs=base_model.input, outputs=predictions)
				if args.gpus is not None:
					model = multi_gpu_model(model, gpus=int(args.gpus))
				# Compile model
				model.compile(optimizer=SGD(lr=0.001, decay=1e-9, momentum=0.9, nesterov=True),
					loss='categorical_crossentropy',
					metrics=['accuracy'])
				# Fit the model
				model.fit(x_train, to_categorical(y_train), batch_size=int(args.batch_size), epochs=args.epochs, verbose=1, callbacks=[tbCallBack], shuffle=True, validation_data=(x_test, to_categorical(y_test)))
				# Measure and print execution time
				end = time.time()
				print('[METRICS] Execution time: {}ms'.format(end-start))
				# Evaluate the model
				scores = model.evaluate(x_test, to_categorical(y_test), verbose=0)
				print('[RESULTS] Accuracy: {}%'.format(scores[1]*100))
			# Save model
			save_model_json(model=model, fich='./cnn_model.json')
			save_weights_hdf5(model=model, fich='./cnn_weights.h5')
		else:
			# Split train and test data
			x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=12345)
			# Load pretrained model
			print('[INFO] Loading InceptionV3 pretrained model...\n')
			model = load_model_json(fich='./cnn_model.json')
			load_weights_hdf5(model=model, fich='./cnn_weights.h5')
			model.compile(optimizer=SGD(lr=0.001, decay=1e-9, momentum=0.9, nesterov=True),
				loss='categorical_crossentropy',
				metrics=['accuracy'])
			print('[INFO] InceptionV3 pretrained model successfully loaded\n')
		# Evaluate the model
		real, pred = [], []
		for i in range(len(x_test)):
			prediction = model.predict(np.array([x_test[i]]))
			real.append(int(y_test[i]))
			pred.append(int(np.argmax(prediction[0], axis=-1)))
		# print(real)
		# print(pred)
		# Plot ROC curve
		fpr = dict()
		tpr = dict()
		roc_auc = dict()
		fpr['dl'], tpr['dl'], _ = roc_curve(np.array(real), np.array(pred))
		roc_auc['dl'] = auc(fpr['dl'], tpr['dl'])
		plt.figure()
		lw = 1
		plt.plot(fpr['dl'], tpr['dl'], color='blue',
				 lw=lw, label='DL Model + FC Layers (area = %0.3f)' % roc_auc['dl'])
		plt.plot([0, 1], [0, 1], color='black', lw=lw, linestyle='--')
		# Check if classifier for feature classification
		if args.classifier == 'SVM':
			# Create model with base model outputs for feature extraction
			model = Model(inputs=model.input, outputs=model.get_layer("dense_2").output)
			# Support Vector Machines Classification
			if not os.path.isfile(os.path.join(os.getcwd(), args.PATH, '.svm')):
				# Extract features from CNN network
				print('[INFO] Extracting features...\n')
				features = []
				labels = []
				for i in range(len(x_train)):
					features_temp = model.predict(np.array([x_train[i]]))
					features.append(features_temp.flatten())
					labels.append(y_train[i])
				# Fit SVM model to the features extracted
				print('[INFO] Training SVM model...\n')
				clf = SVC()
				clf.fit(features, labels)
				serialized = pickle.dumps(clf)
				with open(os.path.join(os.getcwd(), args.PATH, '.svm'), "wb") as f:
					f.write(serialized)
			else:
				print('[INFO] Loading pretrained SVM model...\n')
				with open(os.path.join(os.getcwd(), args.PATH, '.svm'), "rb") as f:
					serialized = f.read()
				clf = pickle.loads(serialized)
			# Model Evaluation
			print('[INFO] Evaluating model...\n')
			real, pred = [], []
			for i in range(len(x_test)):
				features_temp = model.predict(np.array([x_test[i]]))
				features = features_temp.flatten()
				real.append(y_test[i])
				prediction = clf.predict([features])
				pred.append(prediction[0])
			print('[INFO] Accuracy: {}\n'.format(accuracy_score(real, pred)))
			# Plot ROC curve
			fpr['svm'], tpr['svm'], _ = roc_curve(np.array(real), np.array(pred))
			roc_auc['svm'] = auc(fpr['svm'], tpr['svm'])
			plt.plot(fpr['svm'], tpr['svm'], color='red',
					 lw=lw, label='DL Model + SVM (area = %0.3f)' % roc_auc['svm'])
			plt.xlim([0.0, 1.0])
			plt.ylim([0.0, 1.05])
			plt.xlabel('False Positive Rate')
			plt.ylabel('True Positive Rate')
			plt.legend(loc="lower right")
			plt.show()
		else:
			plt.xlim([0.0, 1.0])
			plt.ylim([0.0, 1.05])
			plt.xlabel('False Positive Rate')
			plt.ylabel('True Positive Rate')
			plt.legend(loc="lower right")
			plt.show()
