# Suppress python warnings about deprecations
import warnings
warnings.simplefilter("ignore")

# Suppress tensorflow logging information
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Imports
import cv2
import numpy as np
import tensorflow as tf

def global_contrast_normalization(img, s, lmda, epsilon):
    # Get mean value from the image
    img_average = np.mean(img)
    # Apply filter
    img = img - img_average
    contrast = np.sqrt(lmda + np.mean(img**2))
    img = s * img / max(contrast, epsilon)
    return img

print("OpenCV Version: {0}".format(cv2.__version__))
print("Tensorflow Version: {0}".format(tf.__version__))

# Read image
img = cv2.imread('cnn-architecture.png', 0)

# Show original image
cv2.imshow('Original', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Apply global contrast normalization
img = global_contrast_normalization(img, 1, 10, 0.000000001)

# Show image
cv2.imshow('Global Constrast Normalization', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
