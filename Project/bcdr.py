import os
import csv
import cv2 as cv
import numpy as np
import tensorflow as tf

from tensorflow.python.framework import dtypes
from tensorflow.python.framework.ops import convert_to_tensor

class Reader(object):
    '''
    Wrapper class around the new Tensorflows dataset pipeline.
    Requires Tensorflow >= version 1.12rc0
    '''

    def __init__(self, path, csv, mode, batch_size, num_classes, shuffle=True, buffer_size=1000, save=False):
        '''
        Create a new BCDR Tensorflow instance.
        Recieves a path string to a csv file, which consists of many lines,
            where each line has a path string to an image, patient id, study id,
            lesion outline points and a classification. Using this data,
            this class will create Tensorflow datasets, that can be used to train
            e.g. a convolutional neural network.
        Args:
            csv_file: Path to the text file.
            mode: Either 'training' or 'validation'. Depending on this value,
                different parsing functions will be used.
            batch_size: Number of images per batch.
            num_classes: Number of classes in the dataset.
            shuffle: Wether or not to shuffle the data in the dataset and the
                initial file list.
            buffer_size: Number of images used as buffer for TensorFlows
                shuffling of the dataset.
        Raises:
            ValueError: If an invalid mode is passed.
        '''
        self.path = path
        self.csv = csv
        self.save = save
        self.num_classes = num_classes
        # Retrieve the data from the text file
        self._read_csv_file()
        # Number of samples in the dataset
        self.data_size = len(self.labels)
        # Initial shuffling of the file and label lists (together!)
        if shuffle:
            self._shuffle_lists()
        # Convert lists to TF tensor
        self.img_paths = convert_to_tensor(self.img_paths, dtype=dtypes.string)
        self.labels = convert_to_tensor(self.labels, dtype=dtypes.int32)
        self.min_x = convert_to_tensor(self.min_x, dtype=dtypes.int32)
        self.min_y = convert_to_tensor(self.min_y, dtype=dtypes.int32)
        self.max_x = convert_to_tensor(self.max_x, dtype=dtypes.int32)
        self.max_y = convert_to_tensor(self.max_y, dtype=dtypes.int32)
        # Create dataset
        data = tf.data.Dataset.from_tensor_slices((self.img_paths, self.labels,
                                                   self.min_x, self.min_y,
                                                   self.max_x, self.max_y))
        # Distinguish between train/infer. when calling the parsing functions
        if mode == 'training':
            data = data.map(self._parse_function_train)
        elif mode == 'inference':
            data = data.map(self._parse_function_inference)
        else:
            raise ValueError("Invalid mode '%s'." % (mode))
        # Shuffle the first `buffer_size` elements of the dataset
        if shuffle:
            data = data.shuffle(buffer_size=buffer_size)
        # Create a new dataset with batches of images
        data = data.batch(batch_size)
        self.data = data

    def _read_csv_file(self):
        '''
        Read the content of the csv file and store it into lists.
        '''
        self.img_paths = []
        self.labels = []
        self.min_x = []
        self.min_y = []
        self.max_x = []
        self.max_y = []
        with open(os.path.join(self.path, self.csv), 'r') as raw_data:
            outlines_reader = csv.DictReader(raw_data, delimiter=',')
            for row in outlines_reader:
                path = row['image_filename'][1:]
                self.img_paths.append(os.path.join(self.path, path))
                self.labels.append(int(row['classification'][1:] == 'Malign'))
                if row['classification'][1:] == 'Malign':
                    color = (255,0,0)
                elif row['classification'][1:] == 'Benign':
                    color = (0,255,0)
                else:
                    print('Error in patient with id {}, study {}. Continuing...'.format(row['patient_id'], row['study_id']))
                    continue
                # Get lesion bounding points
                x_points = np.fromstring(row['lw_x_points'], sep=' ')
                y_points = np.fromstring(row['lw_y_points'], sep=' ')
                # Save with bouding points?
                if self.save:
                    img = cv.imread(os.path.join(self.path, path))
                    for i in range (0, x_points.size-2):
                        cv.line(img, (int(x_points[i]), int(y_points[i])), (int(x_points[i+1]), int(y_points[i+1])), color, 3)
                    cv.line(img, (int(x_points[x_points.size-1]), int(y_points[x_points.size-1])), (int(x_points[0]), int(y_points[0])), color, 3)
                    cv.imwrite(os.path.join(self.path, '/bounded/', row['image_filename'][1:]), img)
                # Get bounding box [y,x]
                self.min_x.append(int(min(x_points)-10))
                self.min_y.append(int(min(y_points)-10))
                self.max_x.append(int(max(x_points)+10))
                self.max_y.append(int(max(y_points)+10))

    def _shuffle_lists(self):
        '''
        Conjoined shuffling of the list of paths and labels.
        '''
        path = self.img_paths
        labels = self.labels
        permutation = np.random.permutation(self.data_size)
        self.img_paths = []
        self.labels = []
        for i in permutation:
            self.img_paths.append(path[i])
            self.labels.append(labels[i])

    def _parse_function_train(self, filename, label, min_x, min_y, max_x, max_y):
        '''
        Input parser for samples of the training set.
        '''
        # Convert label number into one-hot-encoding
        one_hot = tf.one_hot(label, self.num_classes)
        # Load and preprocess the image
        img_string = tf.read_file(filename)
        img_decoded = tf.image.decode_png(img_string, channels=3)
        img_resized = tf.image.resize_images(img_decoded, [227, 227])
        # img = cv.imread(os.path.join(self.path, filename))
        # Crop ROI
        # img = img[min_y:max_y, min_x:max_x]
        # ... global contrast norm
        # ... local contrast norm
        # ... resize
        return img_resized, one_hot

    def _parse_function_inference(self, filename, label, min_x, min_y, max_x, max_y):
        '''
        Input parser for samples of the validation/test set.
        '''
        # Convert label number into one-hot-encoding
        one_hot = tf.one_hot(label, self.num_classes)
        # Load and preprocess the image
        img_string = tf.read_file(filename)
        img_decoded = tf.image.decode_png(img_string, channels=3)
        img_resized = tf.image.resize_images(img_decoded, [227, 227])
        # img = cv.imread(os.path.join(self.path, filename))
        # Crop ROI
        # img = img[min_y:max_y, min_x:max_x]
        # ... global contrast norm
        # ... local contrast norm
        # ... resize
        return img_resized, one_hot

    def _global_contrast_normalization(img, s, lmda, epsilon):
        '''
        Global constrast normalization image filter.
        '''
        # Get mean value from the image
        img_average = np.mean(img)
        # Apply filter
        img = img - img_average
        contrast = np.sqrt(lmda + np.mean(img**2))
        img = (s * img) / max(contrast, epsilon)
        return img

    def _local_contrast_normalization(img):
        '''
        Local contrast normalization image filter.
        '''
        pass

    def _data_augmentation(img):
        '''
        Data augmentation process.
        Returns 4 versions of the input image, rotated in 90, 180 and 270 degrees.
        '''
        pass
