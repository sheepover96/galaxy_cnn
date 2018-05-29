import pandas as pd
import numpy as np
from PIL import Image
from astropy.io import fits
import csv

import os
import sys

DATA_ROOT_DIR = '/Users/sheep/Documents/research/project/hsc'
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET = 'dropout_test.csv'
SAVE_DIR = '/Users/sheep/Documents/research/project/hsc/png_images/'
SAVE_FILE_NAME = 'dropout_png.csv'

IMG_CHANNEL = 4

ID_IDX = 0
IMG_IDX = 2
LABEL_IDX = IMG_CHANNEL + IMG_IDX

IMG_SAVE = True


def normalize(image):
    min_value = image.min()
    if min_value < 0:
        image = image - min_value
        min_value = 0
    max_value = image.max()
    #max_value = image.max()
    #normalized = (image - min_value + max_value/20.0).astype(float)*255 / (max_value - min_value + max_value/20.0).astype(float)
    normalized = (image - min_value).astype(float)*255 / (max_value - min_value).astype(float)
    normalized = np.clip(normalized, normalized.min(), 255)
    #print("min = %s, max = %s" % (normalized.min(), normalized.max()))
    return normalized


def zoom_img(img, original_size, pickup_size):
    startpos = int(original_size / 2) - int(pickup_size / 2)
    img = img[startpos:startpos+pickup_size, startpos:startpos+pickup_size]
    return img


def combine_images(images):
    (rows, cols) = (images[0].shape[0], images[0].shape[1])
    combined_image = np.zeros((rows, cols, IMG_CHANNEL))
    for i in range(0, rows):
        for j in range(0, cols):
            for k in range(0, IMG_CHANNEL):
                combined_image[i, j, k] = images[k][i, j]
    #combined_image = np.array([img for img in images]).transpose(1,2,0)
    return combined_image


def load_image(img_paths):
    img_path_list = [ DATA_ROOT_DIR + img_path for img_path in img_paths ]
    image_list = []
    for filepath in img_path_list:
        hdulist = fits.open(filepath)
        row_data = hdulist[0].data
        if row_data is None:
            row_data = hdulist[1].data
        #row_data = zoom_img(row_data, row_data.shape[0], 50)
        image_list.append(row_data)
    #image = combine_images(image_list)
    image = np.array([img for img in image_list]).transpose(1,2,0)
    #print(image_path_list)
    #plt.imshow(image)
    #plt.show()
    #image = normalize(image)
    image = Image.fromarray(np.uint8(image))
    return image

if __name__ == '__main__':
    out_data_frame = pd.DataFrame()
    data_frame = pd.read_csv(os.path.join(ROOT_DIR, 'dataset', DATASET))
    with open(os.path.join(ROOT_DIR, 'dataset', SAVE_FILE_NAME), 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        for idx, data_line in data_frame.iterrows():
            img_id = data_line[ID_IDX]
            img_name = data_line[ID_IDX+1]
            label = data_line[LABEL_IDX]
            img_name = str(label) + '_' + str(img_name) + '.png'
            img_paths = data_line[IMG_IDX:IMG_IDX+IMG_CHANNEL]
            row_data = [img_id, img_name]
            img_paths = [path for path in img_paths]
            row_data.extend(img_paths)
            row_data.append(label)
            writer.writerow(row_data)
            if IMG_SAVE:
                image = load_image(img_paths)
                image.save(os.path.join(SAVE_DIR, img_name))
