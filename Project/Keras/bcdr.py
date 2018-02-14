import os
import csv
import cv2 as cv
import numpy as np

class Reader(object):
    '''
    Wrapper class around the new Tensorflows dataset pipeline.
    Requires Tensorflow >= version 1.12rc0
    '''

    def __init__(self, path, csv, num_classes, shuffle=True, save=False):
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
        # Retrieve the data from the csv file
        self._read_csv_file()

    def _read_csv_file(self):
        '''
        Read the content of the csv file and store it into lists.
        '''
        self.data = np.array([])
        self.labels = np.array([])
        with open(os.path.join(self.path, self.csv), 'r') as raw_data:
            outlines_reader = csv.DictReader(raw_data, delimiter=',')
            for row in outlines_reader:
                path = os.path.join(self.path, row['image_filename'][1:])
                img = cv.imread(path)
                # Benign => green
                # Malign => red
                if row['classification'][1:] == 'Benign':
                    color = (0, 255, 0)
                    self.labels = np.append(self.labels, [0])
                elif row['classification'][1:] == 'Malign':
                    color = (255, 0, 0)
                    self.labels = np.append(self.labels, [1])
                else:
                    print('Error on study {} from patient with id {}, ignoring...'.format(row['study_id'], row['patient_id']))
                    continue
                # Get lesion bounding points
                x_points = np.fromstring(row['lw_x_points'], sep=' ')
                y_points = np.fromstring(row['lw_y_points'], sep=' ')
                if self.save == True:
                    for i in range (0, x_points.size-2):
                        cv.line(img, (int(x_points[i]), int(y_points[i])), (int(x_points[i+1]), int(y_points[i+1])), color, 3)
                    cv.line(img, (int(x_points[x_points.size-1]), int(y_points[x_points.size-1])), (int(x_points[0]), int(y_points[0])), color, 3)
                    cv.imwrite('lines/' + row['image_filename'][1:], img)
                # Get bounding box [y,x]
                min_x = int(min(x_points)-10)
                min_y = int(min(y_points)-10)
                max_x = int(max(x_points)+10)
                max_y = int(max(y_points)+10)
                # cv.rectangle(img, (min_x, min_y), (max_x, max_y), color, 5)
                roi_img = img[min_y:max_y, min_x:max_x]
                try:
                    cv.resize(roi_img, (32, 32))
                    if self.save == True:
                        cv.imwrite('cropped/' + row['image_filename'][1:], roi_img)
                    self.data = np.append(self.data, [roi_img])
                except:
                    pass

    def load_data(self):
        size = len(self.data)
        print(size)
        x_train = self.data[0:int(0.66*size)]
        y_train = self.labels[0:int(0.66*size)]
        x_test = self.data[int(0.66*size):size]
        y_test = self.labels[int(0.66*size):size]
        return (x_train, y_train), (x_test, y_test)
