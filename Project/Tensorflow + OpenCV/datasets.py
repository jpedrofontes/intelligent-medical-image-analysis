import os
import csv
import sys
import json
import pickle
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
            print('\n[INFO] Using BCDR default instance ({})'.format(instance))
            path = os.path.join(current_path, instance)
        else:
            if os.path.isdir(os.path.join(current_path, instance)):
                print('\n[INFO] Using {} instance'.format(instance))
                path = os.path.join(current_path, instance)
            else:
                print('\n[ERROR] The is no instance available of the BCDR dataset with that name.')
                for dir in dirs:
                    print('\t- {}'.format(dir))
                sys.exit()
        # Check if already cached
        if os.path.isdir(os.path.join(current_path, instance, 'cache')):
            print('\n[INFO] {} instance already cached, skipping...\n'.format(instance))
            # Deserialize object
            i=0
            dataset = []
            names = ['x_train', 'y_train', 'x_test', 'y_test']
            for name in names:
                with open(os.path.join(current_path, instance, 'cache', names[i]), "rb") as f:
                    serialized = f.read()
                i = i+1
                deserialized = pickle.loads(serialized)
                dataset.append(deserialized)
            return (dataset[0], dataset[1]), (dataset[2], dataset[3])
        else:
            print('\n[INFO] Processing {} instance...\n'.format(instance))
            os.makedirs(os.path.join(current_path, instance, 'cache'))
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
                        print (roi_img.shape)
                        images.append(roi_img)
                        labels.append(label)
                    except:
                        pass
            images = np.array(images, dtype=np.float32)
            labels = np.array(labels, dtype=np.float32)
            size = len(images)
            division = int(0.66*size)
            # Separate data
            x_train = images[0:division]
            y_train = labels[0:division]
            x_test = images[division+1:size]
            y_test = labels[division+1:size]
            # Shuffle training data
            # shuffle = np.random.randint(low=0, high=x_train.shape[0])
            # Serialize object
            i=0
            names = ['x_train', 'y_train', 'x_test', 'y_test']
            for array in [x_train, y_train, x_test, y_test]:
                serialized = pickle.dumps(array, protocol=0)
                with open(os.path.join(current_path, instance, 'cache', names[i]), "wb") as f:
                    f.write(serialized)
                i+=1
            return ((x_train), y_train), (x_test, y_test)
