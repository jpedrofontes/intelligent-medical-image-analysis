import os
import csv
import sys
import cv2 as cv
import numpy as np

def load_data(instance = None):
    # Check BCDR instance to use
    current_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    if instance is None:
        instance = 'BCDR-F01'
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
    data = []
    labels = []
    with open(os.path.join(path, 'outlines.csv'), 'r') as raw_data:
        outlines_reader = csv.DictReader(raw_data, delimiter=',')
        for row in outlines_reader:
            img_path = os.path.join(path, row['image_filename'][1:])
            img = cv.imread(img_path)
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
            if save == True:
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
                cv.resize(roi_img, (32,32))
                if save == True:
                    cv.imwrite('cropped/' + row['image_filename'][1:], roi_img)
                data.append(roi_img)
                labels.append(label)
            except:
                pass
    data = np.asarray(data)
    labels = np.asarray(labels)
    size = data.shape[0]
    # Separate data
    x_train = data[0:int(0.66*size)]
    y_train = labels[0:int(0.66*size)]
    x_test = data[int(0.66*size):size]
    y_test = labels[int(0.66*size):size]
    # Shuffle training data
    shuffle = np.random.permutation(x_train.shape[0])
    return (x_train[shuffle], y_train[shuffle]), (x_test, y_test)
