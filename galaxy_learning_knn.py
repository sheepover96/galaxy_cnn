
from PIL import Image

import os
import sys
import csv
import random
import base64
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torchvision import transforms


FILE_HOME = "/Users/sheep/Documents/research/project/hsc"
DATA_ROOT_DIR = '/home/okura/research/project/hsc'

DATASET = 'dropout_png.csv'


class DatasetLoader:

    def __init__(self, csv_file_path, root_dir):
        data_frame = pd.read_csv(csv_file_path, header=None)
        self.dataset_frame_list = []
        self.dataset = []
        for i in range(CLASS_NUM):
            if i == 1:
                tmp_dataframe = data_frame[data_frame[LABEL_IDX]==i]
                self.dataset_frame_list.append(tmp_dataframe.sample(n=5000))
            else:
                self.dataset_frame_list.append(data_frame[data_frame[LABEL_IDX]==i])
            self.dataset.append( self.create_dataset(i) )

    def create_dataset(self, label):
        imgCrop = transforms.CenterCrop(IMG_SIZE)
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
            image = imgCrop(image)
            image = np.array(image)

            data_list.append( (label, image, img_no, png_img_name, img_names) )

        return data_list

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


if __name__ == '__main__':

    dataset = DatasetLoader(argv[1], DATA_ROOT_DIR)
    true_dataset = dataset.get_dataset(1)
    false_dataset = dataset.get_dataset(0)

    kfold = KFold(n_splits=5)

    true_dataset_fold = kfold.split(true_dataset)
    false_dataset_fold = kfold.split(false_dataset)
