import os
import csv
import sys
import json
import pickle
import skimage
import numpy as np
import matplotlib.pyplot as plt

from skimage import data, transform, io
from skimage.color import rgb2gray

def global_contrast_normalization(X, s, lmda, epsilon):
    # replacement for the loop
    X_average = np.mean(X)
    X = X - X_average
    contrast = np.sqrt(lmda + np.mean(X**2))
    X = s * X / max(contrast, epsilon)
    return X

class bcdr:
    """
    docstring for BCDR.
    """
    F01 = 'BCDR-F01'
    F02 = 'BCDR-F02'
    F03 = 'BCDR-F03'
    D01 = 'BCDR-D01'
    D02 = 'BCDR-D02'
    DN01 = 'BCDR-DN01'

    @staticmethod
    def load_data(instance = None, save_rois=False, target_size=(32, 32, 3)):
        # Check BCDR instance to use
        current_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
        if instance is None:
            instance = F01
            print('\n[INFO] Using BCDR default instance ({}).'.format(instance))
            path = os.path.join(current_path, instance)
        else:
            if os.path.isdir(os.path.join(current_path, instance)):
                print('\n[INFO] Using {} instance.'.format(instance))
                path = os.path.join(current_path, instance)
            else:
                print('\n[ERROR] The is no instance available of the BCDR dataset with that name.')
                for dir in dirs:
                    print('\t- {}'.format(dir))
        # Check if already cached
        if os.path.isdir(os.path.join(current_path, instance, '.cache')):
            print('\n[INFO] {} instance already cached, skipping.\n'.format(instance))
            # Deserialize object
            i=0
            dataset = []
            names = ['images', 'labels']
            for name in names:
                with open(os.path.join(current_path, instance, '.cache', names[i]), "rb") as f:
                    serialized = f.read()
                i = i+1
                deserialized = pickle.loads(serialized)
                dataset.append(deserialized)
            return (dataset[0], dataset[1])
        else:
            print('\n[INFO] Processing {} instance...\n'.format(instance))
            num_classes = 2
            save = False
            # Retrieve the data from the csv file
            images = []
            labels = []
            with open(os.path.join(path, 'outlines.csv'), 'r') as raw_data:
                outlines_reader = csv.DictReader(raw_data, delimiter=',')
                for row in outlines_reader:
                    img_path = os.path.join(path, row['image_filename'][1:])
                    img = io.imread(img_path)
                    # Benign => green
                    # Malign => red
                    if row['classification'][1:] == 'Benign':
                        color = (0, 255, 0)
                        label = 0
                    elif row['classification'][1:] == 'Malign':
                        color = (255, 0, 0)
                        label = 1
                    else:
                        print('Error on study {} from patient with id {} => ignored.'.format(row['study_id'], row['patient_id']))
                        continue
                    # Get lesion bounding points
                    x_points = np.fromstring(row['lw_x_points'], sep=' ')
                    y_points = np.fromstring(row['lw_y_points'], sep=' ')
                    # Get bounding box [y,x]
                    min_x = int(min(x_points))
                    min_y = int(min(y_points))
                    max_x = int(max(x_points))
                    max_y = int(max(y_points))
                    try:
                        roi_img = img[min_y:max_y, min_x:max_x]
                        roi_img = rgb2gray(roi_img)
                        roi_img = np.stack((roi_img,) * 3, -1)
                        roi_img = transform.resize(roi_img, target_size)
                        images.append(global_contrast_normalization(roi_img, 1, 10, 0.000000001))
                        labels.append(label)
                    except:
                        pass
            # Data Augmentation Process
            for i in range(len(images)):
                # 90 degrees rotation
                img = transform.rotate(images[i], angle=90)
                images.append(img)
                labels.append(labels[i])
                # 180 degrees rotation
                img = transform.rotate(images[i], angle=180)
                images.append(img)
                labels.append(labels[i])
                # 270 degrees rotation
                img = transform.rotate(images[i], angle=-90)
                images.append(img)
                labels.append(labels[i])
                # Flipped images
                img_flip = np.fliplr(images[i])
                images.append(img_flip)
                labels.append(labels[i])
                # 90 degrees rotation
                img = transform.rotate(img_flip, angle=90)
                images.append(img)
                labels.append(labels[i])
                # 180 degrees rotation
                img = transform.rotate(img_flip, angle=180)
                images.append(img)
                labels.append(labels[i])
                # 270 degrees rotation
                img = transform.rotate(img_flip, angle=270)
                images.append(img)
                labels.append(labels[i])
            # Final Numpy Array
            images = np.array(images)
            labels = np.array(labels)
            # Serialize object
            i=0
            names = ['images', 'labels']
            os.makedirs(os.path.join(current_path, instance, '.cache'))
            for array in [images, labels]:
                serialized = pickle.dumps(array)
                with open(os.path.join(current_path, instance, '.cache', names[i]), "wb") as f:
                    f.write(serialized)
                i+=1
            if save_rois:
                if not os.path.isdir(os.path.join(current_path, instance, 'ROIs')):
                    print('[INFO] Saving ROI\'s extracted...\n', )
                    os.mkdir(os.path.join(current_path, instance, 'ROIs'))
                    os.mkdir(os.path.join(current_path, instance, 'ROIs/benign'))
                    os.mkdir(os.path.join(current_path, instance, 'ROIs/malign'))
                    for i in range(images.shape[0]):
                        if labels[i] == 0:
                            io.imsave(os.path.join(current_path, instance, 'ROIs/benign', '{}.png'.format(i)), images[i])
                        else:
                            io.imsave(os.path.join(current_path, instance, 'ROIs/malign', '{}.png'.format(i)), images[i])
            return (images, labels)
