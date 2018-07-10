import torchvision
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.utils.data import ConcatDataset, DataLoader, sampler

from PIL import Image
from skimage import io, transform
from sklearn.model_selection import train_test_split, KFold
from statistics import mean
import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from astropy.io import fits

import os
import sys


GPU = torch.cuda.is_available()

TRUE_DATA_NUM = 12263
FALSE_DATA_NUM = 263

if not GPU:
    DATA_ROOT_DIR = '/Users/sheep/Documents/research/project/hsc'
else:
    DATA_ROOT_DIR = '/home/okura/research/project/hsc'
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET = 'dropout_png.csv'
SAVE_DIR = ''
RESULT_FILE_PATH = 'galaxy_cnn/result/'
RESULT_FILE = 'predict_result.csv'

if not GPU:
    PNG_IMG_DIR = '/Users/sheep/Documents/research/project/hsc/png_images'
else:
    PNG_IMG_DIR = '/home/okura/research/project/hsc/png_images'

IMG_CHANNEL = 4
IMG_SIZE = 50

BATCH_SIZE = 10
NEPOCH = 100
KFOLD = 5

TEST_RATE = 0.2
CLASS_NUM = 2

IMG_IDX = 2
LABEL_IDX = IMG_CHANNEL + IMG_IDX

PNG_LABEL_IDX = 2 + IMG_CHANNEL

SAVE_MODE = True


DATA_TYPE = 'png'


class SNDataset(Dataset):

    def __init__(self, csv_file_path, root_dir, label, transform=None):
        self.img_dataframe = pd.read_csv(csv_file_path, header=None)
        self.root_dir = root_dir
        self.transform = transform

    def __getitem__(self,index):
        img0_tuple = random.choice(self.imageFolderDataset.imgs)
        #we need to make sure approx 50% of images are in the same class
        should_get_same_class = random.randint(0,1)
        if should_get_same_class:
            while True:
                #keep looping till the same class image is found
                img1_tuple = random.choice(self.imageFolderDataset.imgs)
                if img0_tuple[1]==img1_tuple[1]:
                    break
        else:
            while True:
                #keep looping till a different class image is found

                img1_tuple = random.choice(self.imageFolderDataset.imgs)
                if img0_tuple[1] !=img1_tuple[1]:
                    break
            img1_tuple = random.choice(self.imageFolderDataset.imgs)

        img0 = Image.open(img0_tuple[0])
        img1 = Image.open(img1_tuple[0])
        img0 = img0.convert("L")
        img1 = img1.convert("L")

        if self.should_invert:
            img0 = PIL.ImageOps.invert(img0)
            img1 = PIL.ImageOps.invert(img1)

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)

        return img0, img1 , torch.from_numpy(np.array([int(img1_tuple[1]!=img0_tuple[1])],dtype=np.float32))

    def __len__(self):
        return len(self.imageFolderDataset.imgs)

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.cnn1 = nn.Sequential(
            nn.Conv2d(IMG_CHANNEL, 64, kernel_size=10),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ReflectionPad2d(1),
            nn.Conv2d(4, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),

            nn.ReflectionPad2d(1),
            nn.Conv2d(8, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),
        )

        self.fc1 = nn.Sequential(
            nn.Linear(8*100*100, 500),
            nn.ReLU(inplace=True),

            nn.Linear(500, 500),
            nn.ReLU(inplace=True),

            nn.Linear(500, 5)
            )

    def forward_once(self, x):
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2


class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss_contrastive


def train(epoch, model, optimizer, train_loader):
    model.train()
    correct = 0
    total = 0
    for batch_idx, (img_id, img_name, img_names, image, label) in enumerate(train_loader):
        if GPU:
            image, label = image.cuda(), label.cuda()
        image, label = Variable(image), Variable(label)
        optimizer.zero_grad()
        output = model(image)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        if batch_idx % BATCH_SIZE == BATCH_SIZE - 1:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(image), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data.item()))


def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    conf_matrix = [ [ 0 for j in range(CLASS_NUM) ] for i in range(CLASS_NUM)]
    for (img_id, img_name, img_names, image, label) in test_loader:
        with torch.no_grad():
            if GPU:
                image, label = image.cuda(), label.cuda()
            image, label = Variable(image.float()), Variable(label)
            output = model(image)
            test_loss += criterion(output, label).data.item() # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            flatten_label = label.data.view_as(pred)
            correct += pred.eq(flatten_label).long().cpu().sum()
            for i in range(CLASS_NUM):
                for j in range(CLASS_NUM):
                    conf_matrix[i][j] += pred[flatten_label==i].eq(j).long().cpu().sum().item()


if __name__ == '__main__':

    input_file_path = os.path.join(ROOT_DIR, 'dataset', DATASET)

    if DATA_TYPE == 'png':
        ImageDataset = PngImageDataset
    else:
        ImageDataset = FitsImageDataset


    for split in range(5):
        start = split * ( TRUE_DATA_NUM//5 )
        end = (split + 1) * ( TRUE_DATA_NUM//5 )
        print('start:', start)
        print('end:', end)
        true_img_dataset = ImageDataset(input_file_path, DATA_ROOT_DIR, 1, transform=transforms.Compose([
            transforms.CenterCrop(IMG_SIZE),
            transforms.ToTensor(),
            #transforms.Normalize((10,), (50,))
            ]), start=start, end=end)

        false_img_dataset = ImageDataset(input_file_path, DATA_ROOT_DIR, 0, transform=transforms.Compose([
            transforms.CenterCrop(IMG_SIZE),
            #transforms.RandomHorizontalFlip(),
            #transforms.RandomVerticalFlip(),
            #transforms.RandomRotation(300),
            transforms.ToTensor(),
            #transforms.RandomAffine(180, translate=(10, 10)),
            #transforms.Normalize((0.1307,), (0.3081,))
            ]))

        #false data augumentation
        tf_combinations = get_transform_combination2()
        for tf in tf_combinations:
            tf1 = []
            tf1.extend(tf)
            tf1.append(transforms.CenterCrop(IMG_SIZE))
            tf1.append(transforms.ToTensor())
            false_aug = ImageDataset(input_file_path, DATA_ROOT_DIR, 0, transform=transforms.Compose(
                tf1
            ))
            false_img_dataset = ConcatDataset([false_img_dataset, false_aug])

        kfold = KFold(n_splits=KFOLD)

        true_dataset_fold = kfold.split(true_img_dataset)
        false_dataset_fold = kfold.split(false_img_dataset)
        accuracy = []

        #model training and test prediction with k fold cross validation
        for fold_idx, ( (true_train_idx, true_test_idx), (false_train_idx, false_test_idx) ) in\
                enumerate( zip(true_dataset_fold, false_dataset_fold) ):

            true_train_data = [true_img_dataset[i] for i in true_train_idx]
            true_test_data = [true_img_dataset[i] for i in true_test_idx]
            false_train_data = [false_img_dataset[i] for i in false_train_idx]
            false_test_data = [false_img_dataset[i] for i in false_test_idx]

            #image data for prediction
            pr_true_test_data = [true_img_dataset[i] for i in true_test_idx]
            pr_false_test_data = [false_img_dataset[i] for i in false_test_idx]

            train_data = ConcatDataset([true_train_data, false_train_data])
            test_data = ConcatDataset([true_test_data, false_test_data])
            pr_test_data = ConcatDataset([pr_true_test_data, pr_false_test_data])

            train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
            test_loader = DataLoader(test_data, batch_size=100, shuffle=True)
            pr_test_loader = DataLoader(pr_test_data, batch_size=100, shuffle=False)

            print('N TRUE TRAIN: {}\nN TRUE TEST: {}'.format(len(true_train_data), len(true_test_data)))
            print('N FALSE TRAIN: {}\nN FALSE TEST: {}'.format(len(false_train_data), len(false_test_data)))

            model = Net()
            if GPU:
                model.cuda()
            optimizer = optim.Adam(model.parameters(), lr=0.0001)

            test_acc = []
            test_loss = []

            for epoch in range(1, NEPOCH + 1):
                train(epoch, model, optimizer, train_loader)
                tloss, acc = test(model, test_loader)
                test_loss.append(tloss)
                test_acc.append(acc)

            result_file = '{}_{}_{}'.format(split, fold_idx, RESULT_FILE)
            result_file_path = os.path.join(RESULT_FILE_PATH, result_file)
            acc, matrix = predict(model, pr_test_loader, result_file_path)
            accuracy.append(acc.item())
            for i in range(CLASS_NUM):
                for j in range(CLASS_NUM):
                    print(matrix[i][j], end=', ')
                print('\n')
            fig, (figL, figR) = plt.subplots(ncols=2, figsize=(10,4))
            figL.plot(range(1, 1+NEPOCH), np.array(test_loss), marker='o', linewidth=2)
            figL.set_title('test loss')
            figL.set_xlabel('epoch')
            figL.set_ylabel('loss')
            figL.set_xlim(0, NEPOCH)
            figL.grid(True)

            figR.plot(np.array(range(NEPOCH)), np.array(test_acc), marker='o', linewidth=2)
            figR.set_title('test accuracy')
            figR.set_xlabel('epoch')
            figR.set_ylabel('accuracy')
            figR.set_xlim(0, NEPOCH)
            figR.grid(True)

            graph_name = '{}_{}_result.png'.format(split, fold_idx)

            print('save result image:', graph_name)
            fig.savefig(os.path.join('result', 'graph', graph_name))

        print('mean', accuracy)
        #fig.show()
