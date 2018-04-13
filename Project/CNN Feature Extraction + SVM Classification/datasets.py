import os
import csv
import sys
import json
import math
import pickle
import random
import skimage
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def rotate(img, angle):
    rows, cols, chans = img.shape
    M = cv.getRotationMatrix2D((cols/2,rows/2), angle, 1)
    dst = cv.warpAffine(img, M, (cols,rows))
    return dst

def getEnclosingCircle(points):
	# Convert to float and randomize order
	shuffled = [(float(x), float(y)) for (x, y) in points]
	random.shuffle(shuffled)

	# Progressively add points to circle or recompute circle
	c = None
	for (i, p) in enumerate(shuffled):
		if c is None or not is_in_circle(c, p):
			c = _make_circle_one_point(shuffled[ : i + 1], p)
	return c


# One boundary point known
def _make_circle_one_point(points, p):
	c = (p[0], p[1], 0.0)
	for (i, q) in enumerate(points):
		if not is_in_circle(c, q):
			if c[2] == 0.0:
				c = make_diameter(p, q)
			else:
				c = _make_circle_two_points(points[ : i + 1], p, q)
	return c


# Two boundary points known
def _make_circle_two_points(points, p, q):
	circ = make_diameter(p, q)
	left = None
	right = None
	px, py = p
	qx, qy = q

	# For each point not in the two-point circle
	for r in points:
		if is_in_circle(circ, r):
			continue

		# Form a circumcircle and classify it on left or right side
		cross = _cross_product(px, py, qx, qy, r[0], r[1])
		c = make_circumcircle(p, q, r)
		if c is None:
			continue
		elif cross > 0.0 and (left is None or _cross_product(px, py, qx, qy, c[0], c[1]) > _cross_product(px, py, qx, qy, left[0], left[1])):
			left = c
		elif cross < 0.0 and (right is None or _cross_product(px, py, qx, qy, c[0], c[1]) < _cross_product(px, py, qx, qy, right[0], right[1])):
			right = c

	# Select which circle to return
	if left is None and right is None:
		return circ
	elif left is None:
		return right
	elif right is None:
		return left
	else:
		return left if (left[2] <= right[2]) else right


def make_circumcircle(p0, p1, p2):
	# Mathematical algorithm from Wikipedia: Circumscribed circle
	ax, ay = p0
	bx, by = p1
	cx, cy = p2
	ox = (min(ax, bx, cx) + max(ax, bx, cx)) / 2.0
	oy = (min(ay, by, cy) + max(ay, by, cy)) / 2.0
	ax -= ox; ay -= oy
	bx -= ox; by -= oy
	cx -= ox; cy -= oy
	d = (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by)) * 2.0
	if d == 0.0:
		return None
	x = ox + ((ax * ax + ay * ay) * (by - cy) + (bx * bx + by * by) * (cy - ay) + (cx * cx + cy * cy) * (ay - by)) / d
	y = oy + ((ax * ax + ay * ay) * (cx - bx) + (bx * bx + by * by) * (ax - cx) + (cx * cx + cy * cy) * (bx - ax)) / d
	ra = math.hypot(x - p0[0], y - p0[1])
	rb = math.hypot(x - p1[0], y - p1[1])
	rc = math.hypot(x - p2[0], y - p2[1])
	return (x, y, max(ra, rb, rc))


def make_diameter(p0, p1):
	cx = (p0[0] + p1[0]) / 2.0
	cy = (p0[1] + p1[1]) / 2.0
	r0 = math.hypot(cx - p0[0], cy - p0[1])
	r1 = math.hypot(cx - p1[0], cy - p1[1])
	return (cx, cy, max(r0, r1))


_MULTIPLICATIVE_EPSILON = 1 + 1e-14

def is_in_circle(c, p):
	return c is not None and math.hypot(p[0] - c[0], p[1] - c[1]) <= c[2] * _MULTIPLICATIVE_EPSILON

# Returns twice the signed area of the triangle defined by (x0, y0), (x1, y1), (x2, y2).
def _cross_product(x0, y0, x1, y1, x2, y2):
	return (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0)

def global_contrast_normalization(X, s, lmda, epsilon):
    # Calculate mean of intensities
    X_average = np.mean(X)
    # Subtract mean to every pixel
    X = X - X_average
    # Final operation to make stddev equal to s
    contrast = np.sqrt(lmda + np.mean(X**2))
    X = s * X / max(contrast, epsilon)
    return X

def compute_intensity_gradient(image, min_gradient=0):
    """Given an intensity single-channel image it computes the corresponding
    intensity gradient. The gradient image provides the maximum difference
    between each pixel and its immediate neighbours (in row or column)
    as a 2D image.

    Pixels where the gradient value is lower than min_gradient are set to zero.
    """
    N,M = image.shape
    imWork = image.copy().astype(np.float16)
    gradImag = np.zeros_like(image)

    iWork_rowprev = np.pad(imWork[1:,:] - imWork[:-1,:], ((1,0),(0,0)), 'constant')
    iWork_rownext = np.pad(imWork[:-1,:] - imWork[1:,:], ((0,1),(0,0)), 'constant')

    iWork_colprev = np.pad(imWork[:,1:] - imWork[:,:-1], ((0,0),(1,0)), 'constant')
    iWork_colnext = np.pad(imWork[:,:-1] - imWork[:,1:], ((0,0),(0,1)), 'constant')

    canvas = np.zeros([N,M,4])
    canvas[:,:,0] = np.abs(iWork_rowprev)
    canvas[:,:,1] = np.abs(iWork_rownext)
    canvas[:,:,2] = np.abs(iWork_colprev)
    canvas[:,:,3] = np.abs(iWork_colnext)

    gradImag = np.amax(canvas, axis=-1)
    gradImag[gradImag<min_gradient]=0
    return gradImag

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
    def load_data(instance = None, save_rois=False, target_size=(32, 32)):
        # Check BCDR instance to use
        current_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
        if instance is None:
            instance = F01
            print('\n[INFO] Using BCDR default instance ({}).'.format(instance))
            path = os.path.join(current_path, instance)
        else:
            if os.path.isdir(os.path.join(current_path, instance)):
                path = os.path.join(current_path, instance)
            else:
                print('\n[ERROR] The is no instance available of the BCDR dataset with that name.')
                for dir in dirs:
                    print('\t- {}'.format(dir))
        # Check if already cached
        if os.path.isdir(os.path.join(current_path, instance, '.cache')):
            print('\n[INFO] {} instance already cached, skipping.'.format(instance))
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
            print('\n[INFO] Processing {} instance...'.format(instance))
            num_classes = 2
            save = False
            show = False
            # Retrieve the data from the csv file
            images = []
            labels = []
            with open(os.path.join(path, 'outlines.csv'), 'r') as raw_data:
                outlines_reader = csv.DictReader(raw_data, delimiter=',')
                for row in outlines_reader:
                    try:
                        patient = []
                        pat_lab = []
                        img_path = os.path.join(path, row['image_filename'])
                        img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
                        # Benign => green
                        # Malign => red
                        if row['classification'] == 'Benign':
                            color = (0, 255, 0)
                            label = 0
                        elif row['classification'] == 'Malign':
                            color = (255, 0, 0)
                            label = 1
                        else:
                            print('\nError on study {} from patient with id {} => ignored.'.format(row['study_id'], row['patient_id']))
                            continue
                        # Get lesion bounding points
                        x_points = np.fromstring(row['lw_x_points'], sep=' ')
                        y_points = np.fromstring(row['lw_y_points'], sep=' ')
                        if show:
                            img_tmp = cv.imread(img_path)
                            for i in range (0, x_points.size-2):
                                cv.line(img_tmp, (int(x_points[i]), int(y_points[i])), (int(x_points[i+1]), int(y_points[i+1])), color, 3)
                            cv.line(img_tmp, (int(x_points[x_points.size-1]), int(y_points[x_points.size-1])), (int(x_points[0]), int(y_points[0])), color, 3)
                            fig = plt.figure()
                            fig.add_subplot(1,4,1)
                            plt.imshow(img_tmp, cmap='gray')
                            plt.axis('off')
                        # Get bounding circle
                        cnt = []
                        for x,y in zip(x_points, y_points):
                            cnt.append(np.array([x, y]))
                        (x, y, radius) = getEnclosingCircle(cnt)
                        if radius < (target_size[0]*0.5):
                            radius = math.floor(target_size[0]*0.5)
                        # Bounding square from circle
                        min_x = math.floor(x-radius)
                        min_y = math.floor(y-radius)
                        max_x = math.floor(x+radius)
                        max_y = math.floor(y+radius)
                        width, height = img.shape[:2]
                        if min_x < 0:
                            min_x = 0
                        if min_y < 0:
                            min_y = 0
                        if max_x > height:
                            max_x = height-1
                        if max_y > width:
                            max_y = width-1
                        # Crop image
                        roi_img = img[min_y:max_y, min_x:max_x]
                        if show:
                            fig.add_subplot(1,4,2)
                            plt.imshow(roi_img, cmap='gray')
                            plt.axis('off')
                        # Intensity gradient
                        roi_img = compute_intensity_gradient(roi_img, 2.5)
                        roi_img = roi_img.astype(np.uint8)
                        if show:
                            fig.add_subplot(1,4,3)
                            plt.imshow(roi_img, cmap='gray')
                            plt.axis('off')
                        # roi_img = cv.adaptiveThreshold(roi_img, np.amax(roi_img), cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 11, 2)
                        roi_img = cv.equalizeHist(roi_img)
                        cv.normalize(roi_img, roi_img, 0, 255, cv.NORM_MINMAX)
                        if show:
                            fig.add_subplot(1,4,4)
                            plt.imshow(roi_img, cmap='gray')
                            plt.axis('off')
                            plt.show()
                        roi_img = np.stack((roi_img,)*3, -1)
                        roi_img = cv.resize(roi_img, target_size)
                        # Final ROI
                        patient.append(roi_img)
                        pat_lab.append(label)
                        # 90 degrees rotation
                        img = rotate(roi_img, angle=90)
                        patient.append(img)
                        pat_lab.append(label)
                        # 180 degrees rotation
                        img = rotate(roi_img, angle=180)
                        patient.append(img)
                        pat_lab.append(label)
                        # 270 degrees rotation
                        img = rotate(roi_img, angle=-90)
                        patient.append(img)
                        pat_lab.append(label)
                        # Append cases
                        images.append(patient)
                        labels.append(pat_lab)
                    except:
                        pass
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
                    print('\n[INFO] Saving ROI\'s extracted...', )
                    os.mkdir(os.path.join(current_path, instance, 'ROIs'))
                    os.mkdir(os.path.join(current_path, instance, 'ROIs/benign'))
                    os.mkdir(os.path.join(current_path, instance, 'ROIs/malign'))
                    for i in range(images.shape[0]):
                        for j in range(images.shape[1]):
                            # try:
                                if labels[i][j] == 0:
                                    cv.imwrite(os.path.join(current_path, instance, 'ROIs/benign', '{:04d}-{:04d}.png'.format(i,j)), images[i][j])
                                else:
                                    cv.imwrite(os.path.join(current_path, instance, 'ROIs/malign', '{:04d}-{:04d}.png'.format(i,j)), images[i][j])
                            # except:
                                # pass
            return (images, labels)
