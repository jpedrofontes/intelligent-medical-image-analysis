import os
import csv
import sys
import skimage
import numpy as np

from skimage import data, transform
from skimage.color import rgb2gray

class BCDR:
    """
    docstring for BCDR.
    """
    F01 = 'BCDR-F01'
    F02 = 'BCDR-F02'

    @staticmethod
    def load_data(instance = None):
        # Check BCDR instance to use
        current_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
        if instance is None:
            instance = F01
            print('\n[WARNING] Using BCDR default instance ({})'.format(instance))
            path = os.path.join(current_path, instance)
        else:
            if os.path.isdir(os.path.join(current_path, instance)):
                path = os.path.join(current_path, instance)
            else:
                print('\n[ERROR] The is no instance available of the BCDR dataset with that name.')
                for dir in dirs:
                    print('\t- {}'.format(dir))
                sys.exit()

        num_classes = 2
        save = False

        # Retrieve the data from the csv file
        images = []
        labels = []
        with open(os.path.join(path, 'outlines.csv'), 'r') as raw_data:
            outlines_reader = csv.DictReader(raw_data, delimiter=',')
            for row in outlines_reader:
                img_path = os.path.join(path, row['image_filename'][1:])
                img = data.imread(img_path)
                # Benign => green
                # Malign => red
                if row['classification'][1:] == 'Benign':
                    color = (0, 255, 0)
                    label = 0
                elif row['classification'][1:] == 'Malign':
                    color = (255, 0, 0)
                    label = 1
                else:
                    print('Error on study {} from patient with id {}, ignoring...'.format(row['study_id'], row['patient_id']))
                    continue
                # Get lesion bounding points
                x_points = np.fromstring(row['lw_x_points'], sep=' ')
                y_points = np.fromstring(row['lw_y_points'], sep=' ')
                # Get bounding box [y,x]
                min_x = int(min(x_points)-10)
                min_y = int(min(y_points)-10)
                max_x = int(max(x_points)+10)
                max_y = int(max(y_points)+10)
                try:
                    roi_img = img[min_y:max_y, min_x:max_x]
                    roi_img = transform.resize(roi_img, (32, 32))
                    roi_img = rgb2gray(roi_img)
                    images.append(roi_img)
                    labels.append(label)
                except:
                    pass
        images = np.array(images)
        size = len(images)
        division = int(0.66*size)
        # Separate data
        x_train = images[0:division]
        y_train = labels[0:division]
        x_test = images[division+1:size]
        y_test = labels[division+1:size]
        # Shuffle training data
        shuffle = np.random.randint(low=0, high=x_train.shape[0])
        return (x_train, y_train), (x_test, y_test)
