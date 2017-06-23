from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Dense
from keras.layers.core import Dropout
from keras.layers.core import Flatten
from keras.models import Sequential, load_model
from keras.utils import np_utils
from keras.utils.np_utils import to_categorical
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

from astropy.io import fits

import os
import numpy
import sys
import csv
import numpy as np
from pandas import DataFrame
from sklearn.model_selection import train_test_split

class_num = 2 # the number of classes for classification
#input_shape = (1, 239, 239) # ( channels, cols, rows )
#raw_size = (239, 239, 1)
raw_size = (25, 25, 1)
#input_shape = (50, 50, 1)
input_shape = (16, 16, 1)

train_test_split_rate = 0.8
nb_epoch = 20
batch_size = 10
#validation_split = 0.1
validation_split = 0.0

path_to_home = "../"

class DatasetLoader:
    def __init__(self, input_file_path):
        self.train_image_set, self.train_label_set, self.test_image_set, self.test_label_set \
            = self.__import_dataset(input_file_path)

    def zoom_img(self, img, original_size, pickup_size):
        startpos = int(original_size / 2) - int(pickup_size / 2)
        img = img[startpos:startpos+pickup_size, startpos:startpos+pickup_size]
        return img

    def __import_dataset(self, input_file_path):
        dataset = []
        with open(input_file_path, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)
            for row in reader:
                filepath = path_to_home + row[1]
                hdulist = fits.open(filepath)
                raw_image = hdulist[1].data
                image = np.resize(raw_image, [raw_size[0], raw_size[1]])
                image = self.zoom_img(image, raw_size[0], input_shape[0])
                label = int(row[2])
                dataset.append( (label, image) )

        train_image_set = []
        train_label_set = []
        test_image_set = []
        test_label_set = []
        for i in range(0, class_num):
            images = list(map(lambda x: x[1], list(filter(lambda x: x[0] == i, dataset))))
            labels = len(images)*[i]
            train_X, test_X, train_Y, test_Y = train_test_split(images, labels, train_size=train_test_split_rate)
            train_image_set.extend(train_X)
            train_label_set.extend(train_Y)
            test_image_set.extend(test_X)
            test_label_set.extend(test_Y)
        return ( train_image_set, train_label_set, test_image_set, test_label_set )

class GalaxyClassifier:
    def __init__(self):
        self.model = Sequential()
        #self.build_model()

    def build_model_lbg(self):
        self.model.add(Conv2D(10, 3, 3, border_mode='same', input_shape=(input_shape[0], input_shape[1], input_shape[2])))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="tf"))
        self.model.add(Conv2D(64, 3, 3))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="tf"))
        #self.model.add(Dropout(0.25))

        self.model.add(Flatten())
        self.model.add(Dense(256))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(class_num))
        self.model.add(Activation('softmax'))

    def build_model_lae(self):
        self.model.add(Conv2D(10, 3, 3, border_mode='same', input_shape=(input_shape[0], input_shape[1], input_shape[2])))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="tf"))
        #self.model.add(Conv2D(64, 3, 3))
        #self.model.add(Activation('relu'))
        #self.model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="tf"))
        #self.model.add(Dropout(0.25))

        self.model.add(Flatten())
        self.model.add(Dense(16))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(class_num))
        self.model.add(Activation('softmax'))

    def train(self, train_image_set, train_label_set):
        optimizer = Adam(lr=0.001)
        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        train_image_set = np.array(train_image_set)
        print(train_image_set.shape)
        train_image_set = train_image_set.reshape(train_image_set.shape[0], input_shape[0], input_shape[1], input_shape[2])
        train_label_set = to_categorical(train_label_set)
        #early_stopping = EarlyStopping(monitor='val_loss', patience=5)
        #self.model.fit(train_image_set, train_label_set, nb_epoch=20, batch_size=10, validation_split=0.1, callbacks=[early_stopping])
        self.model.fit(train_image_set, train_label_set, nb_epoch=nb_epoch, batch_size=batch_size, validation_split=validation_split)

    def evaluate(self, test_image_set, test_label_set):
        test_image_set = np.array(test_image_set)
        test_label_set = np.array(test_label_set)
        test_image_set = test_image_set.reshape(test_image_set.shape[0], input_shape[0], input_shape[1], input_shape[2])
        test_label_set = to_categorical(test_label_set)
        score = self.model.evaluate(test_image_set, test_label_set, verbose=0)
        print("%s: %.2f%%" % (self.model.metrics_names[1], score[1] * 100))

if __name__ == "__main__":
    argv = sys.argv
    if len(argv) != 2:
        print('Usage: python %s input_file_path' %argv[0])
        quit()
    dataset = DatasetLoader(argv[1])
    galaxyClassifier = GalaxyClassifier()
    galaxyClassifier.build_model_lae()
    galaxyClassifier.train(dataset.train_image_set, dataset.train_label_set)

    galaxyClassifier.evaluate(dataset.test_image_set, dataset.test_label_set)
