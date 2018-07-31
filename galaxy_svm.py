from keras.preprocessing.image import ImageDataGenerator

import csv
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

import itertools
from astropy.io import fits

import os
import sys

CLASS_NUM = 2 # the number of classes for classification

#img_channels = 1
img_channels = 3
IMG_CHANNEL = 3
IMG_SIZE = 50

#input_shape = (1, 239, 239) # ( channels, cols, rows )
raw_size = (239, 239, img_channels)
#raw_size = (48, 48, img_channels)
input_shape = (50, 50, IMG_CHANNEL)
#input_shape = (24, 24, img_channels)

train_test_split_rate = 0.8
#train_test_split_rate = 1
nb_epoch = 20
batch_size = 10
validation_split = 0.1
#validation_split = 0.0


BATCH_SIZE = 10
NEPOCH = 100
KFOLD = 5

IMG_IDX = 2
LABEL_IDX = 2 +  IMG_CHANNEL
PNG_LABEL_IDX = 2 + IMG_CHANNEL

FILE_HOME = "/Users/sheep/Documents/research/project/hsc"

DATA_ROOT_DIR = '/Users/sheep/Documents/research/project/hsc'
ROOT_DIR = os.path.dirname(os.path.abspath(sys.argv[0]))

PNG_IMG_DIR = '/Users/sheep/Documents/research/project/hsc/png_images'
PNG_IMG_DIR = '/Users/sheep/documents/research/project/hsc/png_images'

SAVE_DIR = '/Users/sheep/Documents/research/project/hsc/saved_data'
DATASET = 'dataset/dropout1.csv'

FEATURE_RANGE = [5, 10, 15, 25]
PATTERNS = list(itertools.combinations([i for i in range(IMG_CHANNEL)], 2))


class DatasetLoader:

    def __init__(self, csv_file_path, root_dir, start=1, end=12266):
        data_frame = pd.read_csv(csv_file_path, header=None)
        self.root_dir = root_dir
        self.dataset_frame_list = []
        self.dataset = []
        for i in range(CLASS_NUM):
            if i == 1:
                tmp_dataframe = data_frame[data_frame[LABEL_IDX]==i]
                self.dataset_frame_list.append(tmp_dataframe[start:end])
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

            label = row_data[LABEL_IDX]
            #label = np_utils.to_categorical(label, num_classes=CLASS_NUM)

            image = self.load_image(img_names)
            image = image + 0.5
            image = np.where(image < 0, 0, image)
            image = image * 255 / 3.5
            normalized_image = np.where(image  > 255, 255, image)
            image = self.crop_center(normalized_image, IMG_SIZE, IMG_SIZE)
            #image_feature = self.extract_feature(( normalized_image + 0.5 )*10)
            #print(image_feature, label)

            data_list.append( (label, image, img_no, png_img_name, img_names) )

        return data_list



    def crop_center(self, img,cropx, cropy):
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

    def load_image(self, img_paths):
        image_path_list = [self.root_dir + img_path for img_path in img_paths]
        image_list = []
        for filepath in image_path_list:
            hdulist = fits.open(filepath)
            row_data = hdulist[0].data
            if row_data is None:
                row_data = hdulist[1].data
            image_list.append(row_data)
        image = np.array([img for img in image_list]).transpose(1,2,0)
        return image


def extract_feature(image):
    feature_list = []
    for size in FEATURE_RANGE:
        for (channel1, channel2) in PATTERNS:
            image1 = image[:,:,channel1]
            image2 = image[:,:,channel2]
            feature_list.append(calc_pixel_ratio(image1, image2, size))
    return np.array( feature_list )


def calc_pixel_ratio(image1, image2, size):
    cropped_image1 = crop_center2D(image1, size, size)
    cropped_image2 = crop_center2D(image2, size, size)

    return np.exp(cropped_image1).sum()/np.exp(cropped_image2).sum()


def crop_center2D(img,cropx, cropy):
    y,x = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)
    return img[starty:starty+cropy,startx:startx+cropx]


#create dataset for cross validation
dataset = DatasetLoader(DATASET, DATA_ROOT_DIR, 1, 5000)
print('data read start')
true_dataset = dataset.get_dataset(1)
false_dataset = dataset.get_dataset(0)
print('finished')

other_true_dataset = DatasetLoader(DATASET, DATA_ROOT_DIR, start=5001).get_dataset(1)
other_true_test_img = list(map(lambda data: data[1], other_true_dataset))
other_true_test_label = list(map(lambda data: data[0], other_true_dataset))
other_true_test_catalog_ids_set = list(map(lambda data: data[2], other_true_dataset))
other_true_test_png_img_set = list(map(lambda data: data[3], other_true_dataset))
other_true_test_paths_set = list(map(lambda data: data[4], other_true_dataset))

kfold = KFold(n_splits=5)

true_dataset_fold = kfold.split(true_dataset)
false_dataset_fold = kfold.split(false_dataset)

accuracies = []
for fold_idx, ( (true_train_idx, true_test_idx), (false_train_idx, false_test_idx) ) in\
        enumerate( zip(true_dataset_fold, false_dataset_fold) ):

    print(fold_idx)
    true_train_data = [ true_dataset[idx] for idx in true_train_idx]
    true_test_data = [ true_dataset[idx] for idx in true_test_idx ]
    false_train_data = [ false_dataset[idx] for idx in false_train_idx ]
    false_test_data = [ false_dataset[idx] for idx in false_test_idx ]

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

    true_train_img = list(map(lambda data: extract_feature(data[1]), true_train_data))
    true_train_label = list(map(lambda data: data[0], true_train_data))

    true_test_img = list(map(lambda data: data[1], true_test_data)) + other_true_test_img
    true_test_label = list(map(lambda data: data[0], true_test_data)) + other_true_test_label
    true_test_catalog_ids_set = list(map(lambda data: data[2], true_test_data)) + other_true_test_catalog_ids_set
    true_test_png_img_set = list(map(lambda data: data[3], true_test_data)) + other_true_test_png_img_set
    true_test_paths_set = list(map(lambda data: data[4], true_test_data)) + other_true_test_paths_set

    false_train_img = list(map(lambda data: extract_feature(data[1]), false_train_data))
    false_train_label = list(map(lambda data: data[0], false_train_data))
    false_test_img = list(map(lambda data: extract_feature(data[1]), false_test_data))
    false_test_label = list(map(lambda data: data[0], false_test_data))

    train_img = true_train_img + false_train_img
    train_label = true_train_label + false_train_label
    test_img = true_test_img + false_test_img
    test_label = true_test_label + false_test_label

    print('true train', len(true_train_img))
    print('true test', len(true_test_img))
    print('false train', len(false_train_img))
    print('false test', len(false_test_img))

    #SVM classification
    model = SVC(kernel='rbf')
    model.fit(train_img, train_label)

    pred_result = model.predict(test_img)
    print(metrics.accuracy_score(test_label, pred_result))
    print(metrics.confusion_matrix(test_label, pred_result))
