# Suppress python warnings about deprecations
import warnings
warnings.simplefilter("ignore")

# Suppress tensorflow logging information
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Imports
import csv
import sys
import cv2 as cv
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# OpenCV an Tensorflow versions
print("OpenCV Version: {}".format(cv.__version__))
print("Tensorflow Version: {}".format(tf.__version__))

# Global constrast normalization image filter
def global_contrast_normalization(img, s, lmda, epsilon):
    # Get mean value from the image
    img_average = np.mean(img)
    # Apply filter
    img = img - img_average
    contrast = np.sqrt(lmda + np.mean(img**2))
    img = (s * img) / max(contrast, epsilon)
    return img

# Local contrast normalization
# def local_contrast_normalization(img):
    # ...

# Data augmentation
# def data_augmentation(dataset):
    # ...

# Base ANN layer definition
def conv2d(x, filter_height, filter_width, num_filters, stride_y, stride_x, name,
         padding='SAME', groups=1):
    """
    Create a convolution layer.
    Adapted from: https://github.com/ethereon/caffe-tensorflow
    """
    # Get number of input channels
    input_channels = int(x.get_shape()[-1])
    # Create lambda function for the convolution
    convolve = lambda i, k: tf.nn.conv2d(i, k,
                                         strides=[1, stride_y, stride_x, 1],
                                         padding=padding)
    with tf.variable_scope(name) as scope:
        # Create tf variables for the weights and biases of the conv2d layer
        weights = tf.get_variable('weights', shape=[filter_height,
                                                    filter_width,
                                                    input_channels/groups,
                                                    num_filters])
        biases = tf.get_variable('biases', shape=[num_filters])
    if groups == 1:
        conv2d = convolve(x, weights)
    # In the cases of multiple groups, split inputs & weights and
    else:
        # Split input and weights and convolve them separately
        input_groups = tf.split(axis=3, num_or_size_splits=groups, value=x)
        weight_groups = tf.split(axis=3, num_or_size_splits=groups,
                                 value=weights)
        output_groups = [convolve(i, k) for i, k in zip(input_groups, weight_groups)]
        # Concat the convolved output together again
        conv2d = tf.concat(axis=3, values=output_groups)
    # Add biases
    bias = tf.reshape(tf.nn.bias_add(conv2d, biases), tf.shape(conv2d))
    # Apply relu function
    relu = tf.nn.relu(bias, name=scope.name)
    return relu

def fully_connected(x, num_in, num_out, name, relu=True):
    """
    Create a fully connected layer.
    """
    with tf.variable_scope(name) as scope:
        # Create tf variables for the weights and biases
        weights = tf.get_variable('weights', shape=[num_in, num_out],
                                  trainable=True)
        biases = tf.get_variable('biases', [num_out], trainable=True)
        # Matrix multiply weights and inputs and add bias
        act = tf.nn.xw_plus_b(x, weights, biases, name=scope.name)
    if relu:
        # Apply ReLu non linearity
        relu = tf.nn.relu(act)
        return relu
    else:
        return act

def max_pool(x, filter_height, filter_width, stride_y, stride_x, name,
             padding='SAME'):
    """
    Create a max pooling layer.
    """
    return tf.nn.max_pool(x, ksize=[1, filter_height, filter_width, 1],
                          strides=[1, stride_y, stride_x, 1],
                          padding=padding, name=name)

def lrn(x, radius, alpha, beta, name, bias=1.0):
    """
    Create a local response normalization layer.
    """
    return tf.nn.local_response_normalization(x, depth_radius=radius,
                                              alpha=alpha, beta=beta,
                                              bias=bias, name=name)

def dropout(x, keep_prob):
    """
    Create a dropout layer.
    """
    return tf.nn.dropout(x, keep_prob)

# AlexNet model
class AlexNet(object):
    """
    Implementation of the AlexNet model network.
    """

    def __init__(self, x, keep_prob, num_classes, skip_layer,
                 weights_path='DEFAULT'):
        """
        Create the graph of the AlexNet model.
        Args:
            x: Placeholder for the input tensor.
            keep_prob: Dropout probability.
            num_classes: Number of classes in the dataset.
            skip_layer: List of names of the layer, that get trained from
                scratch
            weights_path: Complete path to the pretrained weight file, if it
                isn't in the same folder as this code
        """
        # Parse input arguments into class variables
        self.X = x
        self.NUM_CLASSES = num_classes
        self.KEEP_PROB = keep_prob
        self.SKIP_LAYER = skip_layer

        if weights_path == 'DEFAULT':
            self.WEIGHTS_PATH = 'bvlc_alexnet.npy'
        else:
            self.WEIGHTS_PATH = weights_path

        # Call the create function to build the computational graph of AlexNet
        self.create()

    def create(self):
        """
        Create the network graph.
        """
        # 1st Layer: conv2d (w ReLu) -> Lrn -> Pool
        conv1 = conv2d(self.X, 11, 11, 96, 4, 4, padding='VALID', name='conv1')
        norm1 = lrn(conv1, 2, 1e-05, 0.75, name='norm1')
        pool1 = max_pool(norm1, 3, 3, 2, 2, padding='VALID', name='pool1')

        # 2nd Layer: conv2d (w ReLu)  -> Lrn -> Pool with 2 groups
        conv2 = conv2d(pool1, 5, 5, 256, 1, 1, groups=2, name='conv2')
        norm2 = lrn(conv2, 2, 1e-05, 0.75, name='norm2')
        pool2 = max_pool(norm2, 3, 3, 2, 2, padding='VALID', name='pool2')

        # 3rd Layer: conv2d (w ReLu)
        conv3 = conv2d(pool2, 3, 3, 384, 1, 1, name='conv3')

        # 4th Layer: conv2d (w ReLu) splitted into two groups
        conv4 = conv2d(conv3, 3, 3, 384, 1, 1, groups=2, name='conv4')

        # 5th Layer: conv2d (w ReLu) -> Pool splitted into two groups
        conv5 = conv2d(conv4, 3, 3, 256, 1, 1, groups=2, name='conv5')
        pool5 = max_pool(conv5, 3, 3, 2, 2, padding='VALID', name='pool5')

        # 6th Layer: Flatten -> fully_connected (w ReLu) -> Dropout
        flattened = tf.reshape(pool5, [-1, 6*6*256])
        fc6 = fully_connected(flattened, 6*6*256, 4096, name='fc6')
        dropout6 = dropout(fc6, self.KEEP_PROB)

        # 7th Layer: fully_connected (w ReLu) -> Dropout
        fc7 = fully_connected(dropout6, 4096, 4096, name='fc7')
        dropout7 = dropout(fc7, self.KEEP_PROB)

        # 8th Layer: fully_connected and return unscaled activations
        self.fc8 = fully_connected(dropout7, 4096, self.NUM_CLASSES, relu=False, name='fc8')

    def load_initial_weights(self, session):
        """
        Load weights from file into network.
        As the weights from http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/
        come as a dict of lists (e.g. weights['conv1'] is a list) and not as
        dict of dicts (e.g. weights['conv1'] is a dict with keys 'weights' &
        'biases') we need a special load function
        """
        # Load the weights into memory
        weights_dict = np.load(self.WEIGHTS_PATH, encoding='bytes').item()

        # Loop over all layer names stored in the weights dict
        for op_name in weights_dict:
            # Check if layer should be trained from scratch
            if op_name not in self.SKIP_LAYER:
                with tf.variable_scope(op_name, reuse=True):
                    # Assign weights/biases to their corresponding tf variable
                    for data in weights_dict[op_name]:
                        # Biases
                        if len(data.shape) == 1:
                            var = tf.get_variable('biases', trainable=False)
                            session.run(var.assign(data))
                        # Weights
                        else:
                            var = tf.get_variable('weights', trainable=False)
                            session.run(var.assign(data))

# Check BCDR instance to use
if len(sys.argv) == 1:
    print('\n[WARNING] Using BCDR default instance ({})'.format('BCDR-F01'))
    os.chdir('BCDR-F01')
else:
    if os.path.isdir(sys.argv[1]):
        os.chdir(sys.argv[1])
    else:
        print('\n[ERROR] The is no instance available of the BCDR dataset with that name.')
        print('\nAvailable instances:')
        dirs = [d for d in os.listdir(os.getcwd()) if os.path.isdir(d)]
        for dir in dirs:
            print('\t- {}'.format(dir))
        sys.exit()
current_path = os.getcwd()
dataset, rois = [], []

# Read Outlines CSV
with open('outlines.csv', 'r') as raw_data:
    outlines_reader = csv.DictReader(raw_data, delimiter=',')
    for row in outlines_reader:
        path = os.path.join(current_path, row['image_filename'][1:])
        img = cv.imread(path)
        # Benign => green
        # Malign => red
        if row['classification'][1:] == 'Benign':
            color = (0, 255, 0)
        elif row['classification'][1:] == 'Malign':
            color = (255, 0, 0)
        else:
            print('Error on study {} from patient with id {}, ignoring...'.format(row['study_id'], row['patient_id']))
            continue
        # Get lesion bounding points
        x_points = np.fromstring(row['lw_x_points'], sep=' ')
        y_points = np.fromstring(row['lw_y_points'], sep=' ')
        # for i in range (0, x_points.size-2):
        #     cv.line(img, (int(x_points[i]), int(y_points[i])), (int(x_points[i+1]), int(y_points[i+1])), color, 3)
        # cv.line(img, (int(x_points[x_points.size-1]), int(y_points[x_points.size-1])), (int(x_points[0]), int(y_points[0])), color, 3)
        # Get bounding box [y,x]
        min_x = int(min(x_points)-10)
        min_y = int(min(y_points)-10)
        max_x = int(max(x_points)+10)
        max_y = int(max(y_points)+10)
        # cv.rectangle(img, (min_x, min_y), (max_x, max_y), color, 5)
        roi_img = img[min_y:max_y, min_x:max_x]
        dataset.append(img)
        rois.append([roi_img, int(row['classification'][1:] == 'Malign')])

# Data visualization
# for i in range(1,10):
#     img = dataset[i]
#     roi_img = rois[i]
#     f = plt.figure()
#     plt.subplot(2, 1, 1)
#     plt.imshow(img)
#     plt.subplot(2, 1, 2)
#     plt.imshow(roi_img)
#     plt.show()

# Learning params
learning_rate = 0.01
num_epochs = 10
batch_size = 128

# Network params
dropout_rate = 0.5
num_classes = 2
train_layers = ['fc8', 'fc7', 'fc6']

# How often we want to write the tf.summary data to disk
display_step = 20

# Path for tf.summary.FileWriter and to store model checkpoints
filewriter_path = "/tmp/finetune_alexnet/tensorboard"
checkpoint_path = "/tmp/finetune_alexnet/checkpoints"

"""
Main Part of the finetuning Script.
"""
# Create parent path if it doesn't exist
if not os.path.isdir(checkpoint_path):
    os.mkdir(checkpoint_path)

# Place data loading and preprocessing on the cpu
with tf.device('/cpu:0'):
    tr_data = ImageDataGenerator(train_file,
                                 mode='training',
                                 batch_size=batch_size,
                                 num_classes=num_classes,
                                 shuffle=True)
    val_data = ImageDataGenerator(val_file,
                                  mode='inference',
                                  batch_size=batch_size,
                                  num_classes=num_classes,
                                  shuffle=False)
    # create an reinitializable iterator given the dataset structure
    iterator = Iterator.from_structure(tr_data.data.output_types,
                                       tr_data.data.output_shapes)
    next_batch = iterator.get_next()

# Ops for initializing the two different iterators
training_init_op = iterator.make_initializer(tr_data.data)
validation_init_op = iterator.make_initializer(val_data.data)

# TF placeholder for graph input and output
x = tf.placeholder(tf.float32, [batch_size, 227, 227, 3])
y = tf.placeholder(tf.float32, [batch_size, num_classes])
keep_prob = tf.placeholder(tf.float32)

# Initialize model
model = AlexNet(x, keep_prob, num_classes, train_layers)

# Link variable to model output
score = model.fc8

# List of trainable variables of the layers we want to train
var_list = [v for v in tf.trainable_variables() if v.name.split('/')[0] in train_layers]

# Op for calculating the loss
with tf.name_scope("cross_ent"):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=score,
                                                                  labels=y))

# Train op
with tf.name_scope("train"):
    # Get gradients of all trainable variables
    gradients = tf.gradients(loss, var_list)
    gradients = list(zip(gradients, var_list))
    # Create optimizer and apply gradient descent to the trainable variables
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = optimizer.apply_gradients(grads_and_vars=gradients)

# Add gradients to summary
for gradient, var in gradients:
    tf.summary.histogram(var.name + '/gradient', gradient)
# Add the variables we train to the summary
for var in var_list:
    tf.summary.histogram(var.name, var)
# Add the loss to summary
tf.summary.scalar('cross_entropy', loss)

# Evaluation op: Accuracy of the model
with tf.name_scope("accuracy"):
    correct_pred = tf.equal(tf.argmax(score, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
# Add the accuracy to the summary
tf.summary.scalar('accuracy', accuracy)
# Merge all summaries together
merged_summary = tf.summary.merge_all()
# Initialize the FileWriter
writer = tf.summary.FileWriter(filewriter_path)
# Initialize an saver for store model checkpoints
saver = tf.train.Saver()
# Get the number of training/validation steps per epoch
train_batches_per_epoch = int(np.floor(tr_data.data_size/batch_size))
val_batches_per_epoch = int(np.floor(val_data.data_size / batch_size))

# Start Tensorflow session
with tf.Session() as sess:
    # Initialize all variables
    sess.run(tf.global_variables_initializer())
    # Add the model graph to TensorBoard
    writer.add_graph(sess.graph)
    # Load the pretrained weights into the non-trainable layer
    model.load_initial_weights(sess)
    print("{} Start training...".format(datetime.now()))
    print("{} Open Tensorboard at --logdir {}".format(datetime.now(),
                                                      filewriter_path))
    # Loop over number of epochs
    for epoch in range(num_epochs):
        print("{} Epoch number: {}".format(datetime.now(), epoch+1))
        # Initialize iterator with the training dataset
        sess.run(training_init_op)
        for step in range(train_batches_per_epoch):
            # get next batch of data
            img_batch, label_batch = sess.run(next_batch)
            # And run the training op
            sess.run(train_op, feed_dict={x: img_batch,
                                          y: label_batch,
                                          keep_prob: dropout_rate})
            # Generate summary with the current batch of data and write to file
            if step % display_step == 0:
                s = sess.run(merged_summary, feed_dict={x: img_batch,
                                                        y: label_batch,
                                                        keep_prob: 1.})
                writer.add_summary(s, epoch*train_batches_per_epoch + step)
        # Validate the model on the entire validation set
        print("{} Start validation".format(datetime.now()))
        sess.run(validation_init_op)
        test_acc = 0.
        test_count = 0
        for _ in range(val_batches_per_epoch):

            img_batch, label_batch = sess.run(next_batch)
            acc = sess.run(accuracy, feed_dict={x: img_batch,
                                                y: label_batch,
                                                keep_prob: 1.})
            test_acc += acc
            test_count += 1
        test_acc /= test_count
        print("{} Validation Accuracy = {:.4f}".format(datetime.now(),
                                                       test_acc))
        print("{} Saving checkpoint of model...".format(datetime.now()))
        # save checkpoint of the model
        checkpoint_name = os.path.join(checkpoint_path,
                                       'model_epoch'+str(epoch+1)+'.ckpt')
        save_path = saver.save(sess, checkpoint_name)
        print("{} Model checkpoint saved at {}".format(datetime.now(), checkpoint_name))
