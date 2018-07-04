from torch.utils.data import Dataset
from torchvision import transforms

from PIL import Image
import numpy as np
import pandas as pd
from glob import glob
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfTransformer
import mahotas as mh
from mahotas.features import surf
from astropy.io import fits

import os
import sys


TRUE_DATA_NUM = 12263

DIMENSION = 50
NCLUSTERS = 15
NITER = 500

picture_category_num = 9
feature_category_num = 512

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
PNG_IMG_DIR = '/Users/sheep/Documents/research/project/hsc/png_images'
DATA_ROOT_DIR = '/Users/sheep/Documents/research/project/hsc'
DATASET = 'dropout_png.csv'

IMG_CHANNEL = 4
IMG_SIZE = 50


IMG_IDX = 2
LABEL_IDX = IMG_CHANNEL + IMG_IDX

PNG_LABEL_IDX = 2 + IMG_CHANNEL

class PngImageDataset(Dataset):

    def __init__(self, csv_file_path, root_dir, label, transform=None, start=0, end=TRUE_DATA_NUM):
        tmp_dataframe = pd.read_csv(csv_file_path, header=None)
        self.image_dataframe = tmp_dataframe[tmp_dataframe[LABEL_IDX] == label]
        if label == 1:
            self.image_dataframe = self.image_dataframe[start:end]#.sample(n=100)
        self.root_dir = root_dir
        self.transform = transform


    def __len__(self):
        return len(self.image_dataframe)


    def load_image(self, img_name):
        img_path = os.path.join(PNG_IMG_DIR, img_name)
        image = Image.open(img_path)
        return image


    def __getitem__(self, idx):
        img_id = self.image_dataframe.iat[idx, 0]
        img_names = self.image_dataframe.iloc[idx, 2:IMG_IDX+IMG_CHANNEL]
        img_names = [ path for path in img_names ]
        img_names = ','.join(img_names)
        img_name = self.image_dataframe.iloc[idx, 1]
        image = self.load_image(img_name)
        label = self.image_dataframe.iat[idx, PNG_LABEL_IDX]

        if self.transform:
            image = self.transform(image)

        image = np.array(image)

        return img_id, img_name, img_names, image, label


def normalize(image):
    min_value = image.min()
    if min_value < 0:
        image = image - min_value
        min_value = 0
    max_value = image.max()
    normalized = (image - min_value).astype(float)*255 / (max_value - min_value).astype(float)
    normalized = np.clip(normalized, normalized.min(), 255)
    return normalized


def load_image(img_path):
    full_img_path = DATA_ROOT_DIR + img_path
    hdulist = fits.open(full_img_path)
    row_data = hdulist[0].data
    if row_data is None:
        row_data = hdulist[1].data
    norm_row_data = normalize(row_data)
    image = Image.fromarray(np.uint8(norm_row_data))
    return image


if __name__ == '__main__':

    #data reading
    input_file_path = os.path.join(ROOT_DIR, 'dataset', DATASET)
    true_img_dataset = PngImageDataset(input_file_path, DATA_ROOT_DIR, 1, transform=transforms.Compose([
        transforms.CenterCrop(IMG_SIZE),
        ]), start=1, end=TRUE_DATA_NUM)

    false_img_dataset = PngImageDataset(input_file_path, DATA_ROOT_DIR, 0, transform=transforms.Compose([
        transforms.CenterCrop(IMG_SIZE)
        ]))

    print(len(true_img_dataset))

    true_imgs = []
    for (img_id, img_name, img_names, image, label) in true_img_dataset:
        true_imgs.append(image)

    true_imgs_np = np.array(true_imgs)
    true_imgs_flat = true_imgs_np.reshape(len(true_imgs_np),-1).astype(np.float64)

    ## dimension reduction by PCA
    sc = StandardScaler()
    true_imgs_std = sc.fit_transform(true_imgs_flat.transpose())
    cov_mat = np.cov(true_imgs_std)
    eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)

    for i in range(50):
        print(eigen_vals[i])

    eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[i]) for i in range(len(eigen_vals))]
    eigen_pairs.sort(key=lambda k: k[0], reverse=True)
    w = np.hstack([ eigen_pairs[i][1][:, np.newaxis] for i in range(DIMENSION) ])

    mini_true_imgs_flat = true_imgs_flat.dot(w)

    km = KMeans(n_clusters=NCLUSTERS, max_iter=NITER)
    result = km.fit(mini_true_imgs_flat)

    labels = result.labels_

    for ( (img_id, img_name, img_names, image, label), cls ) in zip(true_img_dataset, labels):
        os.makedirs(os.path.join(DATA_ROOT_DIR, 'clustering', '1', str(cls), img_name), exist_ok=True)
        img_names = img_names.split(',')
        print(cls, img_name)
        for idx, path in enumerate( img_names ):
            pil_img = load_image(path)
            pil_img.save(os.path.join(DATA_ROOT_DIR,\
                    'clustering', '1', str(cls), img_name, str( idx + 1 ) + '.png'))
