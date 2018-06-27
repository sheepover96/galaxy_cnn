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
from keras.preprocessing.image import ImageDataGenerator

from astropy.io import fits

from PIL import Image

import pydot
import graphviz

import os
import sys
import csv
import random
import base64
import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torchvision import transforms

class_num = 2 # the number of classes for classification

#img_channels = 1
img_channels = 4
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

FILE_HOME = "/Users/sheep/Documents/research/project/hsc"

SAVE_DIR = '/Users/sheep/documents/research/project/hsc/saved_data'
transforms.CenterCrop

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
            #filepath = FILE_HOME + filepath
            #データのロード
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
            #combined_image = np.zeros((rows, cols, img_channels))
            #for i in range(0, rows):
            #    for j in range(0, cols):
            #        for k in range(0, img_channels):
            #            combined_image[i, j, k] = images[k][i, j]
            combined_image = np.array([img for img in images]).transpose(1,2,0)
            print(combined_image.shape)
            return combined_image

        #def normalize(image):
        #    return (image - image.min()).astype(float)*255 / (image.max() - image.min()).astype(float)

        def normalize(image):
            min_value = image.min()
            if min_value < 0:
                image = image - min_value
                min_value = 0
            image_center = self.zoom_img(image, image.shape[0], 5)
            max_value = image_center.max()
            #max_value = image.max()
            #normalized = (image - min_value + max_value/20.0).astype(float)*255 / (max_value - min_value + max_value/20.0).astype(float)
            normalized = (image - min_value).astype(float)*255 / (max_value - min_value).astype(float)
            normalized = np.clip(normalized, normalized.min(), 255)
            #print("min = %s, max = %s" % (normalized.min(), normalized.max()))
            return normalized

        def save_as_image(image, output_path):
            image = normalize(image)
            pil_img = Image.fromarray(numpy.uint8(image))
            pil_img.save(output_path)

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

        with open(input_file_path, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)
            #col_size = len(header)
            col_size = 6
            channel_num = 3
            label_index = 6
            #CSVからの読み込み
            for i, row in enumerate(reader):
                #ラベル抽出
                label = int(row[label_index])
                #正解データ数が5000を超えたら終了
                if len(list(filter(lambda x:x[0]==1, dataset))) >= 5000 and label == 1:
                    continue
                print("No. %s started" % i)
                #画像へのパス
                image_paths = row[2:label_index]
                #画像のカタログID
                catalog_id = row[0]
                if img_channels > 1:
                    #各チャネルの画像をリサイズして合成(RGB形式にする)
                    images = [load_and_resize(FILE_HOME + filepath) for filepath in image_paths]
                    image = combine_images(images)
                    #image = special_median_filter(image, 5)
                else:
                    image = load_and_resize(row[1])
                #image = normalize(image)
                #処理後の画像の保存
                combined_filename = '{0}_{1}.png'.format(label, '_'.join(row[1].split('/')[-2:]).replace('/', '_'))
                combined_img_path = FILE_HOME + '/combined_images/{0}'.format(combined_filename)
                if save_mode:
                    save_as_image(image, combined_img_path)

                dataset.append( (label, image, image_paths, catalog_id, combined_img_path) )

        print("DATASET SIZE = %s" % len(dataset))
        print("TRUE SIZE = %s, FALSE SIZE = %s" % ( len(list(filter(lambda x:x[0]==1, dataset))), len(list(filter(lambda x:x[0]==0, dataset))) ))

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
        datagen = ImageDataGenerator(
            horizontal_flip=True,
            vertical_flip=True)
        #読み込んだデータを学習用，テスト用に分割する
        for i in range(0, class_num):
            images = list(map(lambda x: x[1], list(filter(lambda x: x[0] == i, dataset))))
            image_paths = list(map(lambda x: x[2], list(filter(lambda x: x[0] == i, dataset))))
            catalog_ids = list(map(lambda x: x[3], list(filter(lambda x: x[0] == i, dataset))))
            combined_img_paths = list(map(lambda x: x[4], list(filter(lambda x: x[0] == i, dataset))))
            image_path_zipped = list(zip(images, image_paths, catalog_ids, combined_img_paths))
            labels = len(images)*[i]
            train_X, test_X, train_Y, test_Y = train_test_split(image_path_zipped, labels, train_size=train_test_split_rate)
            #誤判定画像を増やす
            if i == 0:
                train_X_tmp = train_X
                train_X = []
                train_Y = []
                for x in train_X_tmp:
                    image = np.expand_dims(x[0], axis=0)
                    #generator = datagen.flow(image, batch_size=1, save_to_dir=SAVE_DIR, save_prefix='img', save_format='png')
                    generator = datagen.flow(image, batch_size=1, save_prefix='img', save_format='png')
                    for ite in range(19):
                        batch = generator.next()
                        train_X.append( (batch[0], x[1], x[2], x[3]) )
                        train_Y.append(i)
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

            print("Train size for class %s is %s" % (i, len(train_images)))
            print("Test size for class %s is %s" % (i, len(test_images)))

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
        self.model.add(Dense(class_num))
        self.model.add(Activation('softmax'))

    def build_model_dropout(self):
        self.model.add(Conv2D(10, 3, 3, padding='same', input_shape=(input_shape[0], input_shape[1], input_shape[2])))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="tf"))

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
        train_image_set = train_image_set.reshape(train_image_set.shape[0], input_shape[0], input_shape[1], input_shape[2])
        train_label_set = to_categorical(train_label_set)
        #early_stopping = EarlyStopping(monitor='val_loss', patience=5)
        #self.model.fit(train_image_set, train_label_set, nb_epoch=20, batch_size=10, validation_split=0.1, callbacks=[early_stopping])
        return self.model.fit(train_image_set, train_label_set, epochs=nb_epoch, batch_size=batch_size, validation_split=validation_split)

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
    """
    print("Start loading dataset")
    dataset = DatasetLoader(argv[1])
    print("loading finished")
    galaxyClassifier = GalaxyClassifier()
    galaxyClassifier.build_model_lbg()
    #galaxyClassifier.build_model_lae()
    galaxyClassifier.train(dataset.train_image_set, dataset.train_label_set)
    """

    trial_count = 1
    accuracies = []
    for i in range(trial_count):
        dataset = DatasetLoader(argv[1])
        galaxyClassifier = GalaxyClassifier()
        #galaxyClassifier.build_model_lbg()
        galaxyClassifier.build_model_lae()
        #galaxyClassifier.build_model_dropout()
        hist = galaxyClassifier.train(dataset.train_image_set, dataset.train_label_set)

        #galaxyClassifier.visualizeFeatureMaps(2)

        acc = hist.history['acc']
        val_acc = hist.history['val_acc']

        epochs = len(acc)
        #plt.plot(range(epochs), acc, marker='.', label='acc')
        #plt.plot(range(epochs), val_acc, marker='.', label='val_acc')
        #plt.legend(loc='best')
        #plt.grid()
        #plt.xlabel('epoch')
        #plt.ylabel('acc')
        #plt.show()
        #plt.savefig("accuracy.png")

        score = galaxyClassifier.evaluate(dataset.test_image_set, dataset.test_label_set)
        print("%s: %.2f%%" % (galaxyClassifier.model.metrics_names[1], score[1] * 100))

        accuracies.append(float(score[1]))

    print("average accuracy = %s" % (sum(accuracies)/len(accuracies)))

    galaxyClassifier.predictAll(dataset.test_image_set, dataset.test_label_set, dataset.test_image_paths_set, dataset.test_catalog_ids_set, dataset.test_combined_img_path_set)
