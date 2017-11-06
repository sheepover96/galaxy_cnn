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
from keras.utils import plot_model

from astropy.io import fits

from PIL import Image

import pydot
import graphviz

import os
import numpy
import sys
import csv
import random
import base64
import numpy as np
from pandas import DataFrame
from sklearn.model_selection import train_test_split

class_num = 2 # the number of classes for classification

#img_channels = 1
img_channels = 3
#input_shape = (1, 239, 239) # ( channels, cols, rows )
raw_size = (239, 239, img_channels)
#raw_size = (48, 48, img_channels)
input_shape = (50, 50, img_channels)
#input_shape = (24, 24, img_channels)

train_test_split_rate = 0.8
#train_test_split_rate = 1
nb_epoch = 20
batch_size = 10
validation_split = 0.1
#validation_split = 0.0

save_mode = True

class DatasetLoader:
    def __init__(self, input_file_path):
        self.train_image_set, self.train_label_set, self.train_image_paths_set, self.train_catalog_ids_set, self.train_combined_img_path_set, \
        self.test_image_set, self.test_label_set, self.test_image_paths_set, self.test_catalog_ids_set, self.test_combined_img_path_set \
            = self.__import_dataset(input_file_path)

    def zoom_img(self, img, original_size, pickup_size):
        startpos = int(original_size / 2) - int(pickup_size / 2)
        img = img[startpos:startpos+pickup_size, startpos:startpos+pickup_size]
        return img
    
    def __import_dataset(self, input_file_path):
        dataset = []

        def load_and_resize(filepath):
            filepath = "/Users/daiz" + filepath
            hdulist = fits.open(filepath)
            raw_image = hdulist[0].data
            if( raw_image is None ):
                raw_image = hdulist[1].data
            #image = np.resize(raw_image, [raw_size[0], raw_size[1]])
            image = raw_image
            image = self.zoom_img(image, raw_size[0], input_shape[0])
            return image

        def combine_images(images):
            (rows, cols) = (images[0].shape[0], images[0].shape[1])
            combined_image = np.zeros((rows, cols, img_channels))
            for i in range(0, rows):
                for j in range(0, cols):
                    for k in range(0, img_channels):
                        combined_image[i, j, k] = images[k][i, j]
            return combined_image

        def normalize(image):
            return (image - image.min()).astype(float)*255 / (image.max() - image.min()).astype(float)

        def save_as_image(image, output_path):
            image = normalize(image)
            pil_img = Image.fromarray(numpy.uint8(image))
            pil_img.save(output_path)

        with open(input_file_path, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)
            #col_size = len(header)
            col_size = 6
            channel_num = 3
            label_index = 5
            for i, row in enumerate(reader):
                print("No. %s started" % i)    
                label = int(row[label_index])
                image_paths = row[2:5]
                catalog_id = row[0]
                if channel_num > 1:
                    images = [load_and_resize(filepath) for filepath in image_paths]
                    image = combine_images(images)
                else:
                    image = load_and_resize(row[1])
                #image = normalize(image)
                combined_filename = '{0}_{1}.png'.format(label, '_'.join(row[1].split('/')[-2:]).replace('/', '_'))
                combined_img_path = '/Users/daiz/combined_images/{0}'.format(combined_filename)
                if save_mode:
                    save_as_image(image, combined_img_path)
                """
                if os.path.isdir(path):
                    files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
                    images = [load_and_resize(f) for f in files]
                    image = combine_images(images)
                else:
                    image = load_and_resize(path)
                """
                dataset.append( (label, image, image_paths, catalog_id, combined_img_path) )

        print("DATASET SIZE = %s" % len(dataset))

        train_image_set = []
        train_label_set = []
        train_image_paths_set = []
        train_catalog_ids_set = []
        train_combined_img_path_set = []
        test_image_set = []
        test_label_set = []
        test_image_paths_set = []
        test_catalog_ids_set = []
        test_combined_img_path_set = []
        for i in range(0, class_num):
            images = list(map(lambda x: x[1], list(filter(lambda x: x[0] == i, dataset))))
            print(len(images))
            image_paths = list(map(lambda x: x[2], list(filter(lambda x: x[0] == i, dataset))))
            catalog_ids = list(map(lambda x: x[3], list(filter(lambda x: x[0] == i, dataset))))
            combined_img_paths = list(map(lambda x: x[4], list(filter(lambda x: x[0] == i, dataset))))
            image_path_zipped = list(zip(images, image_paths, catalog_ids, combined_img_paths))
            labels = len(images)*[i]
            train_X, test_X, train_Y, test_Y = train_test_split(image_path_zipped, labels, train_size=train_test_split_rate)
            train_images = list(map(lambda x: x[0], train_X))
            train_image_paths = list(map(lambda x: x[1], train_X))
            train_catalog_ids = list(map(lambda x: x[2], train_X))
            train_combined_img_paths = list(map(lambda x: x[3], train_X))
            test_images = list(map(lambda x: x[0], test_X))
            test_image_paths = list(map(lambda x: x[1], test_X))
            test_catalog_ids = list(map(lambda x: x[2], test_X))
            test_combined_img_paths = list(map(lambda x: x[3], test_X))
            train_image_set.extend(train_images)
            train_label_set.extend(train_Y)
            train_image_paths_set.extend(train_image_paths)
            train_catalog_ids_set.extend(train_catalog_ids)
            train_combined_img_path_set.extend(train_combined_img_paths)
            test_image_set.extend(test_images)
            test_label_set.extend(test_Y)
            test_image_paths_set.extend(test_image_paths)
            test_catalog_ids_set.extend(test_catalog_ids)
            test_combined_img_path_set.extend(test_combined_img_paths)
        return ( train_image_set, train_label_set, train_image_paths_set, train_catalog_ids_set, train_combined_img_path_set,
             test_image_set, test_label_set, test_image_paths_set, test_catalog_ids_set, test_combined_img_path_set )

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
        
        """
        self.model.add(Conv2D(32, 3, 3))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="tf"))
        #self.model.add(Dropout(0.25))
        """

        self.model.add(Conv2D(64, 3, 3))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="tf"))

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
        return score
        plot_model(self.model, to_file='model.png')

    def predictAll(self, test_image_set, test_label_set, test_image_paths_set, test_catalog_ids_set, test_combined_img_path_set):
        test_image_set = np.array(test_image_set)
        test_label_set = np.array(test_label_set)
        test_image_set = test_image_set.reshape(test_image_set.shape[0], input_shape[0], input_shape[1], input_shape[2])
        test_label_set_categorical = to_categorical(test_label_set)
        predicted = self.model.predict(test_image_set)
        self.__writeResultToCSV(zip(test_catalog_ids_set, test_image_paths_set, test_combined_img_path_set, test_label_set, predicted), './predict_result.csv')
        for (correct_label, probabilities) in zip(test_label_set, predicted):
            print("correct label = %s, probabilities = [%s, %s]" % (correct_label, probabilities[0], probabilities[1]))


    def __writeResultToCSV(self, zipped_result, output_filepath):
        with open(output_filepath, 'w') as f:
            writer = csv.writer(f)
            for result in zipped_result:
                cat_id = result[0]
                img_paths = result[1]
                combined_img_path = result[2]
                label = result[3]
                float_formatter = lambda x: "%.4f" % x
                probabilities = list(map(float_formatter, result[4]))
                # row should be [cat_id, img1, img2, img3, combined_img, correct_label, [probabilties], answer]
                row = [cat_id, img_paths, combined_img_path, label, probabilities]
                writer.writerow(row)

if __name__ == "__main__":
    argv = sys.argv
    if len(argv) != 2:
        print('Usage: python %s input_file_path' %argv[0])
        quit()
    print("Start loading dataset")
    dataset = DatasetLoader(argv[1])
    print("loading finished")
    galaxyClassifier = GalaxyClassifier()
    galaxyClassifier.build_model_lbg()
    #galaxyClassifier.build_model_lae()
    galaxyClassifier.train(dataset.train_image_set, dataset.train_label_set)

    trial_count = 1
    accuracies = []
    for i in range(trial_count):
        dataset = DatasetLoader(argv[1])
        galaxyClassifier = GalaxyClassifier()
        #galaxyClassifier.build_model_lbg()
        galaxyClassifier.build_model_lae()
        galaxyClassifier.train(dataset.train_image_set, dataset.train_label_set)

        score = galaxyClassifier.evaluate(dataset.test_image_set, dataset.test_label_set)
        print("%s: %.2f%%" % (galaxyClassifier.model.metrics_names[1], score[1] * 100))

        accuracies.append(float(score[1]))

    print("average accuracy = %s" % (sum(accuracies)/len(accuracies)))

    galaxyClassifier.predictAll(dataset.test_image_set, dataset.test_label_set, dataset.test_image_paths_set, dataset.test_catalog_ids_set, dataset.test_combined_img_path_set)
