# Imports
import tensorflow as tf

class SimpleMLP(object):
    """
    docstring for SimpleMLP.
    """
    def __init__(self, img_shape, num_classes):
        self.img_shape = img_shape
        self.num_classes = num_classes

    def createModel(self):
        # Initialize placeholders
        self.x = tf.placeholder(dtype = tf.float32, shape = self.img_shape)
        self.y = tf.placeholder(dtype = tf.int32, shape = [None])

        # Flatten the input data
        images_flat = tf.contrib.layers.flatten(self.x)
        # Fully connected layers
        fc1 = tf.contrib.layers.fully_connected(images_flat, 64, tf.nn.relu)
        fc2 = tf.contrib.layers.fully_connected(fc1, 32, tf.nn.relu)
        fc3 = tf.contrib.layers.fully_connected(fc2, 32, tf.nn.tanh)
        logits = tf.contrib.layers.fully_connected(fc3, self.num_classes, tf.nn.relu)
        # Define a loss function
        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = self.y, logits = logits))
        # Define an optimizer
        self.train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.loss)
        # Convert logits to label indexes
        self.correct_pred = tf.argmax(logits, 1)
        # Define an accuracy metric
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))

class AlexNet(object):
    """
    docstring for AlexNet.
    """
    def __init__(self, img_shape, num_classes):
        pass

    def createModel(self, _X, _weights, _biases, _dropout):
        # Initialize placeholders
        self.x = tf.placeholder(dtype = tf.float32, shape = self.img_shape)
        self.y = tf.placeholder(dtype = tf.int32, shape = [None])

        # Reshape input picture
        _X = tf.reshape(_X, shape=[-1, 32, 32, 1])

        # Convolution Layer
        conv1 = conv2d('conv1', _X, _weights['wc1'], _biases['bc1'])
        # Max Pooling (down-sampling)
        pool1 = max_pool('pool1', conv1, k=2)
        # Apply Normalization
        norm1 = norm('norm1', pool1, lsize=4)
        # Apply Dropout
        norm1 = tf.nn.dropout(norm1, _dropout)

        # Convolution Layer
        conv2 = conv2d('conv2', norm1, _weights['wc2'], _biases['bc2'])
        # Max Pooling (down-sampling)
        pool2 = max_pool('pool2', conv2, k=2)
        # Apply Normalization
        norm2 = norm('norm2', pool2, lsize=4)
        # Apply Dropout
        norm2 = tf.nn.dropout(norm2, _dropout)

        # Convolution Layer
        conv3 = conv2d('conv3', norm2, _weights['wc3'], _biases['bc3'])
        # Max Pooling (down-sampling)
        pool3 = max_pool('pool3', conv3, k=2)
        # Apply Normalization
        norm3 = norm('norm3', pool3, lsize=4)
        # Apply Dropout
        norm3 = tf.nn.dropout(norm3, _dropout)

        # Fully connected layer
        # Reshape conv3 output to fit dense layer input
        dense1 = tf.reshape(norm3, [-1, _weights['wd1'].get_shape().as_list()[0]])
        # Relu activation
        dense1 = tf.nn.relu(tf.matmul(dense1, _weights['wd1']) + _biases['bd1'], name='fc1')
        # Relu activation
        dense2 = tf.nn.relu(tf.matmul(dense1, _weights['wd2']) + _biases['bd2'], name='fc2')
        # Output, class prediction
        out = tf.matmul(dense2, _weights['out']) + _biases['out']
        return out

def conv2d(name, l_input, w, b):
    return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(l_input, w, strides=[1, 1, 1, 1],
                                                  padding='SAME'),b), name=name)

def max_pool(name, l_input, k):
    return tf.nn.max_pool(l_input, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME', name=name)

def norm(name, l_input, lsize=4):
    return tf.nn.lrn(l_input, lsize, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name=name)
