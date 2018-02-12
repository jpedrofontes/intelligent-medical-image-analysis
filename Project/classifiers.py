# Imports
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten

class CNN(object):
    """
    docstring for AlexNet.
    """
    def __init__(self, num_classes, shape):
        super(CNN, self).__init__()
        self.num_classes = num_classes
        self.shape = shape

    def createModel(self):
        # Use Sequential API
        model = Sequential()
        # First convolutional Layer
        model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=self.shape))
        model.add(Conv2D(32, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        # Second convolutional Layer
        model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        # Third convolutional Layer
        model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        # Fully connected layer
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.num_classes, activation='softmax'))
        return model
