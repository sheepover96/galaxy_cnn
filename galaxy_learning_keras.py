from keras.callbacks import EarlyStopping
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation, Dense, Dropout, Flatten
from keras.models import Sequential, load_model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import array_to_img, img_to_array, list_pictures, load_img
from keras.utils import np_utils
from keras.utils import plot_model
from keras.utils.np_utils import to_categorical

from astropy.io import fits

from PIL import Image

import pydot
import graphviz

import csv
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import MinMaxScaler

import os
import sys

CLASS_NUM = 2 # the number of classes for classification

#img_channels = 1
img_channels = 4
IMG_CHANNEL = 4
IMG_SIZE = 50

#input_shape = (1, 239, 239) # ( channels, cols, rows )
raw_size = (239, 239, img_channels)
#raw_size = (48, 48, img_channels)
input_shape = (50, 50, IMG_CHANNEL)
#input_shape = (24, 24, img_channels)

train_test_split_rate = 0.8
#train_test_split_rate = 1
nb_epoch = 1
batch_size = 10
validation_split = 0.1
#validation_split = 0.0


BATCH_SIZE = 10
NEPOCH = 100
KFOLD = 5

IMG_IDX = 2
LABEL_IDX = IMG_CHANNEL + IMG_IDX
PNG_LABEL_IDX = 2 + IMG_CHANNEL

FILE_HOME = "/home/okura/research/project/hsc"

DATA_ROOT_DIR = '/home/okura/research/project/hsc'
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

PNG_IMG_DIR = '/home/okura/research/project/hsc/png_images'

SAVE_DIR = '/Users/sheep/documents/research/project/hsc/saved_data'

save_mode = True

class DatasetLoader:

    def __init__(self, csv_file_path, root_dir):
        data_frame = pd.read_csv(csv_file_path, header=None)
        self.dataset_frame_list = []
        self.dataset = []
        for i in range(CLASS_NUM):
            if i == 1:
                tmp_dataframe = data_frame[data_frame[LABEL_IDX]==i]
                self.dataset_frame_list.append(tmp_dataframe.sample(n=253))
            else:
                self.dataset_frame_list.append(data_frame[data_frame[LABEL_IDX]==i])
            self.dataset.append( self.create_dataset(i) )

    def create_dataset(self, label):
        data_frame = self.get_dataframe(label)
        data_list = []

        for idx, row_data in data_frame.iterrows():
            img_no = str(row_data[0])

            png_img_name = row_data[1]

            img_names = row_data[2:IMG_IDX+IMG_CHANNEL]
            img_names = [ path for path in img_names ]

            label = row_data[PNG_LABEL_IDX]
            #label = np_utils.to_categorical(label, num_classes=CLASS_NUM)

            image = Image.open(os.path.join(PNG_IMG_DIR, png_img_name))
            image = np.array(image)
            image = self.crop_center(image, IMG_SIZE, IMG_SIZE)

            data_list.append( (label, image, img_no, png_img_name, img_names) )

        return data_list

    def crop_center(self, img,cropx,cropy):
        y,x,z = img.shape
        startx = x//2-(cropx//2)
        starty = y//2-(cropy//2)
        return img[starty:starty+cropy,startx:startx+cropx,:]


    def get_dataframe(self, label):
        return self.dataset_frame_list[label]

    def get_dataset(self, label):
        return self.dataset[label]

    def zoom_img(self, img, original_size, pickup_size):
        startpos = int(original_size / 2) - int(pickup_size / 2)
        img = img[startpos:startpos+pickup_size, startpos:startpos+pickup_size]
        return img

    def median_filter(image, ksize):
        # 畳み込み演算をしない領域の幅
        d = int((ksize - 1) / 2)
        h, w = image.shape[0], image.shape[1]

        # 出力画像用の配列（要素は入力画像と同じ）
        dst = image.copy()

        for y in range(d, h - d):
            for x in range(d, w - d):
                # 近傍にある画素値の中央値を出力画像の画素値に設定
                dst[y][x] = np.median(image[y - d:y + d + 1, x - d:x + d + 1])

        return dst

    def special_median_filter(src, ksize):
        # 畳み込み演算をしない領域の幅
        d = int((ksize - 1) / 2)
        h, w, c = src.shape[0], src.shape[1], src.shape[2]

        # 出力画像用の配列（要素は入力画像と同じ）
        dst = src.copy()
        result = src.copy()

        for i in range(c):
            for y in range(d, h - d):
                for x in range(d, w - d):
                    # 近傍にある画素値の中央値を出力画像の画素値に設定
                    dst[y][x][i] = np.median(src[y - d:y + d + 1, x - d:x + d + 1, i])

        means = []
        for i in range(c):
            means.append(np.mean(src[:, :, i] - dst[:, :, i]))

        for i in range(c):
            for y in range(d, h - d):
                for x in range(d, w - d):
                    # 近傍にある画素値の中央値を出力画像の画素値に設定
                    pixel = src[y, x, i]
                    # print(pixel - dst[y, x, i])
                    if pixel == 0 or pixel == 255:
                        result[y, x, i] = dst[y, x, i]
                    elif pixel - dst[y, x, i] > means[i]:
                        result[y, x, i] = dst[y, x, i]
        return result


class GalaxyClassifier:
    def __init__(self):
        self.model = Sequential()
        #self.build_model()

    def build_model_lae(self):
        self.model.add(Conv2D(10, (3, 3), input_shape=(input_shape[0], input_shape[1], input_shape[2]), padding="same"))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="tf"))

        """
        self.model.add(Conv2D(32, 3, 3))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="tf"))
        #self.model.add(Dropout(0.25))
        """

        self.model.add(Conv2D(64, (3, 3)))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="tf"))

        self.model.add(Flatten())
        self.model.add(Dense(16))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(CLASS_NUM))
        self.model.add(Activation('softmax'))


    def train(self, train_image_set, train_label_set):
        optimizer = Adam(lr=0.001)
        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        train_image_set = np.array(train_image_set)
        train_image_set = train_image_set.reshape(train_image_set.shape[0], input_shape[0], input_shape[1], input_shape[2])
        train_label_set = np.array(train_label_set)
        train_label_set = to_categorical(train_label_set)
        #early_stopping = EarlyStopping(monitor='val_loss', patience=5)
        #self.model.fit(train_image_set, train_label_set, nb_epoch=20, batch_size=10, validation_split=0.1, callbacks=[early_stopping])
        print(train_image_set.shape)
        print(train_label_set.shape)
        return self.model.fit(train_image_set, train_label_set, epochs=nb_epoch, batch_size=batch_size, validation_split=validation_split)


    def evaluate(self, test_image_set, test_label_set):
        test_image_set = np.array(test_image_set)
        test_label_set = np.array(test_label_set)
        test_label_set = to_categorical(test_label_set)
        test_image_set = test_image_set.reshape(test_image_set.shape[0], input_shape[0], input_shape[1], input_shape[2])
        score = self.model.evaluate(test_image_set, test_label_set, verbose=0)
        pred = self.model.predict_classes(test_image_set)
        return score, pred
        plot_model(self.model, to_file='model.png')


    def predictAll(self, fold, test_image_set, test_label_set, test_image_paths_set, test_catalog_ids_set, test_combined_img_path_set):
        test_image_set = np.array(test_image_set)
        test_label_set = np.array(test_label_set)
        test_image_set = test_image_set.reshape(test_image_set.shape[0], input_shape[0], input_shape[1], input_shape[2])
        test_label_set_categorical = to_categorical(test_label_set)
        predicted = self.model.predict(test_image_set)
        self.__writeResultToCSV(zip(test_catalog_ids_set, test_image_paths_set, test_combined_img_path_set, test_label_set, predicted), 'result/{}_predict_result.csv'.format(fold))
        #for (correct_label, probabilities) in zip(test_label_set, predicted):
        #    print("correct label = %s, probabilities = [%s, %s]" % (correct_label, probabilities[0], probabilities[1]))


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

    def visualizeFeatureMaps(self, layer_num):
        W = self.model.layers[layer_num].get_weights()[0]
        W = W.transpose(3, 2, 1, 0)
        nb_filter, nb_channel, nb_row, nb_col = W.shape
        plt.figure()
        for i in range(nb_filter):
            im = W[i, 0]
            scaler = MinMaxScaler(feature_range=(0, 255))
            im = scaler.fit_transform(im)
            plt.subplot(4, 8, i+1)
            plt.axis('off')
            plt.imshow(im, cmap='gray')
        plt.show()

if __name__ == "__main__":
    argv = sys.argv
    if len(argv) != 2:
        print('Usage: python %s input_file_path' %argv[0])
        quit()


    #create dataset for cross validation
    dataset = DatasetLoader(argv[1], DATA_ROOT_DIR)
    true_dataset = dataset.get_dataset(1)
    false_dataset = dataset.get_dataset(0)

    kfold = KFold(n_splits=5)

    true_dataset_fold = kfold.split(true_dataset)
    print(true_dataset_fold)
    false_dataset_fold = kfold.split(false_dataset)
    print(false_dataset_fold)

    accuracies = []
    for fold_idx, ( (true_train_idx, true_test_idx), (false_train_idx, false_test_idx) ) in\
            enumerate( zip(true_dataset_fold, false_dataset_fold) ):

        true_train_data = [true_dataset[i] for i in true_train_idx]
        true_test_data = [true_dataset[i] for i in true_test_idx]
        false_train_data = [false_dataset[i] for i in false_train_idx]
        false_test_data = [false_dataset[i] for i in false_test_idx]

        #data augumentation
        datagen = ImageDataGenerator(
            horizontal_flip=True,
            vertical_flip=True)

        tmp_false_train_data = false_train_data
        false_train_data = []
        for idx, data in enumerate( tmp_false_train_data ):
            label = data[0]
            img = data[1]
            img_no = data[2]
            img_name = data[3]
            img_names = data[4]
            expanded_image = np.expand_dims(img, axis=0)
            generator = datagen.flow(expanded_image, batch_size=1, save_prefix='img', save_format='png')
            for ite in range(19):
                batch = generator.next()
                false_train_data.append( (label, batch[0], img_no, img_name, img_names) )

        true_train_img = list(map(lambda data: data[1], true_train_data))
        true_train_label = list(map(lambda data: data[0], true_train_data))

        true_test_img = list(map(lambda data: data[1], true_test_data))
        true_test_label = list(map(lambda data: data[0], true_test_data))
        true_test_catalog_ids_set = list(map(lambda data: data[2], true_test_data))
        true_test_png_img_set = list(map(lambda data: data[3], true_test_data))
        true_test_paths_set = list(map(lambda data: data[4], true_test_data))

        false_train_img = list(map(lambda data: data[1], false_train_data))
        false_train_label = list(map(lambda data: data[0], false_train_data))

        false_test_img = list(map(lambda data: data[1], false_test_data))
        false_test_label = list(map(lambda data: data[0], false_test_data))
        false_test_catalog_ids_set = list(map(lambda data: data[2], true_test_data))
        false_test_png_img_set = list(map(lambda data: data[3], true_test_data))
        false_test_paths_set = list(map(lambda data: data[4], true_test_data))

        print('TRUE TRAIN', len(true_train_img))
        print('TRUE TEST', len(true_test_img))
        print('FALSE TRAIN', len(false_train_img))
        print('FALSE TRAIN', len(false_test_img))

        train_img = true_train_img + false_train_img
        train_label = true_train_label + false_train_label
        test_img = true_test_img + false_test_img
        test_label = true_test_label + false_test_label
        test_catalog_ids_set = true_test_catalog_ids_set + false_test_catalog_ids_set
        test_png_img_set = true_test_png_img_set + false_test_png_img_set
        test_paths_set = true_test_paths_set + false_test_paths_set

        accracies = []
        galaxyClassifier = GalaxyClassifier()
        galaxyClassifier.build_model_lae()
        hist = galaxyClassifier.train(train_img, train_label)

        #galaxyClassifier.visualizeFeatureMaps(2)

        acc = hist.history['acc']
        val_acc = hist.history['val_acc']
        loss = hist.history['loss']
        val_loss = hist.history['val_loss']

        epochs = len(acc)
        plt.plot(range(epochs), acc, marker='.', label='acc')
        plt.plot(range(epochs), val_acc, marker='.', label='val_acc')
        plt.legend(loc='best')
        plt.grid()
        plt.xlabel('epoch')
        plt.ylabel('acc')
        #plt.show()
        plt.savefig("{}_kaccuracy.png".format(fold_idx))

        plt.plot(range(epochs), acc, marker='.', label='loss')
        plt.plot(range(epochs), val_acc, marker='.', label='val_loss')
        plt.legend(loc='best')
        plt.grid()
        plt.xlabel('epoch')
        plt.ylabel('loss')
        #plt.show()
        plt.savefig("{}_kloss.png".format(fold_idx))

        score, pred = galaxyClassifier.evaluate(test_img, test_label)
        print("%s: %.2f%%" % (galaxyClassifier.model.metrics_names[1], score[1] * 100))
        print('confusion matrix')
        print(metrics.confusion_matrix(test_label, pred))

        accuracies.append(float(score[1]))

        print("average accuracy = %s" % (sum(accuracies)/len(accuracies)))

        galaxyClassifier.predictAll(
                fold_idx, test_img, test_label,
                test_paths_set, test_catalog_ids_set,
                test_png_img_set
                )
