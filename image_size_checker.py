from astropy.io import fits

import os
import numpy
import sys
import csv
import numpy as np
from pandas import DataFrame

class_num = 2 # the number of classes for classification

img_channels = 3
#input_shape = (1, 239, 239) # ( channels, cols, rows )
raw_size = (239, 239, 1)
#raw_size = (48, 48, img_channels)
input_shape = (50, 50, 1)
#input_shape = (24, 24, img_channels)

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

        def load_and_resize(filepath):
            hdulist = fits.open(filepath)
            raw_image = hdulist[0].data
            if( raw_image == None ):
                raw_image = hdulist[1].data
            print("raw size = {0}, {1}".format(raw_image.shape[0], raw_image.shape[1]))
            image = np.resize(raw_image, [raw_size[0], raw_size[1]])
            #print("size = {0}, {1}".format(image.shape[0], image.shape[1]))
            #image = self.zoom_img(image, raw_size[0], input_shape[0])
            #return image
            return raw_image

        def combine_images(images):
            (rows, cols) = (images[0].shape[0], images[0].shape[1])
            combined_image = np.zeros((rows, cols, img_channels))
            for i in range(0, rows):
                for j in range(0, cols):
                    for k in range(0, channels):
                        combined_image[i, j, k] = images[k][i, j]
            return combined_image

        with open(input_file_path, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)
            for row in reader:
                #label = int(row[2])
                label = int(row[3])
                #path = path_to_home + row[1]
                images = [load_and_resize(path_to_home+filepath) for filepath in row[0:3]]
                #image = combine_images(images)
                """
                if os.path.isdir(path):
                    files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
                    images = [load_and_resize(f) for f in files]
                    image = combine_images(images)
                else:
                    image = load_and_resize(path)
                """

if __name__ == "__main__":
    argv = sys.argv
    if len(argv) != 2:
        print('Usage: python %s input_file_path' %argv[0])
        quit()
    dataset = DatasetLoader(argv[1])
