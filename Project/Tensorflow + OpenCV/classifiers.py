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
        fc2 = tf.contrib.layers.fully_connected(images_flat, 64, tf.nn.relu)
        fc3 = tf.contrib.layers.fully_connected(images_flat, 64, tf.nn.relu)
        fc4 = tf.contrib.layers.fully_connected(images_flat, 64, tf.nn.relu)
        fc5 = tf.contrib.layers.fully_connected(images_flat, 64, tf.nn.relu)
        logits = tf.contrib.layers.fully_connected(fc5, self.num_classes, tf.nn.relu)
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
        self.img_shape = img_shape
        self.num_classes = num_classes

    def createModel(self):
        # Initialize placeholders
        self.x = tf.placeholder(dtype = tf.float32, shape = self.img_shape)
        self.y = tf.placeholder(dtype = tf.int32, shape = [None])

        input_layer = tf.reshape(self.x, [-1, 32, 32, 1])

        # Convolutional layer #1
        conv1 = tf.layers.conv2d(inputs=input_layer, filters=96, kernel_size=[11,11], padding="SAME", activation=tf.nn.relu)
        lrn1 = tf.nn.local_response_normalization(input=conv1, depth_radius=2, alpha=1e-05, beta=0.75)
        pool1 = tf.layers.max_pooling2d(inputs=lrn1, pool_size=[3, 3], strides=2)
        # Convolutional layer #2
        conv2 = tf.layers.conv2d(inputs=pool1, filters=256, kernel_size=[5, 5], padding="SAME", activation=tf.nn.relu)
        lrn2 = tf.nn.local_response_normalization(input=conv2, depth_radius=2, alpha=1e-05, beta=0.75)
        pool2 = tf.layers.max_pooling2d(inputs=lrn2, pool_size=[3, 3], strides=2)
        # Convolutional layer #3
        conv3 = tf.layers.conv2d(inputs=pool2, filters=384, kernel_size=[3, 3], padding="SAME", activation=tf.nn.relu)
        # Convolutional layer #4
        conv4 = tf.layers.conv2d(inputs=conv3, filters=384, kernel_size=[3, 3], padding="SAME", activation=tf.nn.relu)
        # Convolutional layer #5
        conv5 = tf.layers.conv2d(inputs=conv4, filters=256, kernel_size=[3, 3], padding="SAME", activation=tf.nn.relu)
        pool3 = tf.layers.max_pooling2d(inputs=conv5, pool_size=[3, 3], strides=2)
        # Flatten the input data
        flatten = tf.contrib.layers.flatten(pool1)
        # Fully connected layers
        fc1 = tf.contrib.layers.fully_connected(flatten, 4096, tf.nn.relu)
        logits = tf.contrib.layers.fully_connected(fc1, self.num_classes, tf.nn.relu)
        # Define a loss function
        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = self.y, logits = logits))
        # Define an optimizer
        self.train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.loss)
        # Convert logits to label indexes
        self.correct_pred = tf.argmax(logits, 1)
        # Define an accuracy metric
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))

class CNN(object):
    """
    docstring for CNN.
    """
    def __init__(self, img_shape, num_classes):
        self.img_shape = img_shape
        self.num_classes = num_classes

    @staticmethod
    def createModel(features, labels, mode):
        # Input Layer
        input_layer = tf.reshape(features["x"], [-1, 32, 32, 1])
        # Convolutional Layer #1
        conv1 = tf.layers.conv2d(
            inputs=input_layer,
            filters=32,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu)
        # Pooling Layer #1
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
        # Convolutional Layer #2 and Pooling Layer #2
        conv2 = tf.layers.conv2d(
            inputs=pool1,
            filters=64,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu)
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
        # Dense Layer
        pool2_flat = tf.reshape(pool2, [-1, 8*8*64])
        dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
        dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)
        # Logits Layer
        logits = tf.layers.dense(inputs=dropout, units=10)
        predictions = {
            # Generate predictions (for PREDICT and EVAL mode)
            "classes": tf.argmax(input=logits, axis=1),
            # Add `softmax_tensor` to the graph. It is used for PREDICT and by the `logging_hook`.
            "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
        }
        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
        # Calculate Loss (for both TRAIN and EVAL modes)
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
        # Configure the Training Op (for TRAIN mode)
        if mode == tf.estimator.ModeKeys.TRAIN:
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
            train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
        # Add evaluation metrics (for EVAL mode)
        eval_metric_ops = {
            "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])
        }
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
