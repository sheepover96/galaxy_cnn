import torchvision
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.utils.data import ConcatDataset, DataLoader, sampler

import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from statistics import mean
from PIL import Image

from astropy.io import fits

import os
import sys

DATA_ROOT_DIR = '/Users/sheep/Documents/research/project/hsc'
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET = 'dropout2.csv'
SAVE_DIR = '/Users/sheep/Documents/research/project/hsc/saved_data'

IMG_CHANNEL = 4
IMG_SIZE = 50

BATCH_SIZE = 10
NEPOCH = 20
KFOLD = 5

TEST_RATE = 0.2
CLASS_NUM = 2

IMG_IDX = 2
LABEL_IDX = IMG_CHANNEL + IMG_IDX

SAVE_MODE = True

class ImageDataset(Dataset):

    def __init__(self, csv_file_path, root_dir, label, transform=None):
        tmp_dataframe = pd.read_csv(csv_file_path, header=None)
        self.image_dataframe = tmp_dataframe[tmp_dataframe[LABEL_IDX] == label]
        if label == 1:
            self.image_dataframe = self.image_dataframe.sample(n=5000)
        self.root_dir = root_dir
        self.transform = transform


    def __len__(self):
        return len(self.image_dataframe)


    def load_image(self, img_paths):
        image_path_list = [self.root_dir + img_path for img_path in img_paths]
        image_list = []
        for filepath in image_path_list:
            hdulist = fits.open(filepath)
            row_data = hdulist[0].data
            if row_data is None:
                row_data = hdulist[1].data
            image_list.append(row_data)
        #image = np.array([img for img in image_list]).transpose(1,2,0)
        #print(image_path_list)
        #plt.imshow(image)
        #plt.show()
        image = np.array([img for img in image_list]).transpose(1,2,0)
        image = Image.fromarray(np.uint8(image))
        return image


    def get(self, idx):
        img_id = self.image_dataframe.iat[idx, 0]
        img_names = self.image_dataframe.iloc[idx, IMG_IDX:IMG_IDX+IMG_CHANNEL]
        img_names = [ path for path in img_names ]
        image = self.load_image(img_names)
        label = self.image_dataframe.iat[idx, LABEL_IDX]

        if self.transform:
            image = self.transform(image)

        return img_id, img_names, image, label


    def __getitem__(self, idx):

        label = self.image_dataframe.iat[idx, LABEL_IDX]
        img_names = self.image_dataframe.iloc[idx, IMG_IDX:IMG_IDX+IMG_CHANNEL]
        image = self.load_image(img_names)

        if self.transform:
            image = self.transform(image)

        return image, label


def get_transform_combination():
    tr = []
    tr.append([transforms.RandomHorizontalFlip(1)])
    tr.append([transforms.RandomHorizontalFlip(1), transforms.RandomVerticalFlip(1)])
    tr.append([transforms.RandomHorizontalFlip(1), transforms.RandomAffine(180, translate=(0.4, 0.4))])
    tr.append([transforms.RandomHorizontalFlip(1), transforms.RandomRotation(300)])
    tr.append([transforms.RandomHorizontalFlip(1), transforms.RandomVerticalFlip(1), transforms.RandomAffine(180, translate=(0.4, 0.4))])
    tr.append([transforms.RandomHorizontalFlip(1), transforms.RandomVerticalFlip(1), transforms.RandomRotation(300)])
    tr.append([transforms.RandomHorizontalFlip(1), transforms.RandomVerticalFlip(1), transforms.RandomRotation(300), transforms.RandomAffine(180, translate=(0.4, 0.4))])
    tr.append([transforms.RandomHorizontalFlip(1), transforms.RandomAffine(180, translate=(0.4, 0.4)), transforms.RandomRotation(300)])
    tr.append([transforms.RandomVerticalFlip(1)])
    tr.append([transforms.RandomVerticalFlip(1), transforms.RandomAffine(180, translate=(0.4, 0.4))])
    tr.append([transforms.RandomVerticalFlip(1), transforms.RandomRotation(300)])
    tr.append([transforms.RandomVerticalFlip(1), transforms.RandomAffine(180, translate=(0.4, 0.4)), transforms.RandomRotation(300)])
    tr.append([transforms.RandomAffine(180, translate=(0.4, 0.4))])
    tr.append([transforms.RandomAffine(180, translate=(0.4, 0.4)), transforms.RandomRotation(300)])
    tr.append([transforms.RandomRotation(300)])
    return tr


def get_transform_combination2():
    tr = []
    tr.append([transforms.RandomRotation(45, 45)])
    tr.append([transforms.RandomRotation(90, 90)])
    tr.append([transforms.RandomRotation(135, 135)])
    tr.append([transforms.RandomRotation(180, 180)])
    tr.append([transforms.RandomRotation(225, 225)])
    tr.append([transforms.RandomRotation(270, 270)])
    tr.append([transforms.RandomRotation(315, 315)])
    tr.append([transforms.RandomHorizontalFlip(1)])
    return tr


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(IMG_CHANNEL, 10, kernel_size=3)
        self.conv2 = nn.Conv2d(10, 64, kernel_size=3)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(7744, 16)
        self.fc2 = nn.Linear(16, 2)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, 7744)
        x = F.relu(self.fc1(x))
        x = self.conv2_drop(x)
        x = self.fc2(x)
        return F.sigmoid(x)


criterion = nn.CrossEntropyLoss()


def train(epoch, model, optimizer, train_loader):
    model.train()
    correct = 0
    total = 0
    for batch_idx, (image, label) in enumerate(train_loader):
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
    for (image, label) in test_loader:
        with torch.no_grad():
            image, label = Variable(image.float()), Variable(label)
            output = model(image)
            test_loss += criterion(output, label).data.item() # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            flatten_label = label.data.view_as(pred)
            correct += pred.eq(flatten_label).long().cpu().sum()
            for i in range(CLASS_NUM):
                for j in range(CLASS_NUM):
                    conf_matrix[i][j] += pred[flatten_label==i].eq(j).long().cpu().sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), accuracy))
    print(conf_matrix)

    return test_loss, accuracy


def write_result(img_ids, img_names, images, labels):
    pass


def predict(model, test_loader):
    model.eval()
    correct = 0
    conf_matrix = [ [ 0 for j in range(CLASS_NUM) ] for i in range(CLASS_NUM)]
    for (img_id, img_names, image, label) in test_loader:
        image, label = Variable(image.float(), volatile=True), Variable(label)
        output = model(image)
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        flatten_label = label.data.view_as(pred)
        correct += pred.eq(flatten_label).long().cpu().sum()
        print(label, pred)
        for i in range(CLASS_NUM):
            for j in range(CLASS_NUM):
                conf_matrix[i][j] += pred[flatten_label==i].eq(j).long().cpu().sum().item()

    accuracy = 100. * ( correct / len(test_loader.dataset) )
    print(conf_matrix)

    return accuracy, conf_matrix


if __name__ == '__main__':

    input_file_path = os.path.join(ROOT_DIR, 'dataset', DATASET)

    true_img_dataset = ImageDataset(input_file_path, DATA_ROOT_DIR, 1, transform=transforms.Compose([
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        #transforms.Normalize((10,), (50,))
        ]))

    false_img_dataset = ImageDataset(input_file_path, DATA_ROOT_DIR, 0, transform=transforms.Compose([
        transforms.CenterCrop(IMG_SIZE),
        #transforms.RandomHorizontalFlip(),
        #transforms.RandomVerticalFlip(),
        #transforms.RandomRotation(300),
        transforms.ToTensor(),
        #transforms.RandomAffine(180, translate=(10, 10)),
        #transforms.Normalize((0.1307,), (0.3081,))
        ]))

    kfold = KFold(n_splits=KFOLD)

    true_dataset_fold = kfold.split(true_img_dataset)
    false_dataset_fold = kfold.split(false_img_dataset)
    accuracy = []
    for (true_train_idx, true_test_idx), (false_train_idx, false_test_idx) in\
            zip(true_dataset_fold, false_dataset_fold):

        true_train_data = [true_img_dataset[i] for i in true_train_idx]
        true_test_data = [true_img_dataset[i] for i in true_test_idx]
        false_train_data = [false_img_dataset[i] for i in false_train_idx]
        false_test_data = [false_img_dataset[i] for i in false_test_idx]

        tf_combinations = get_transform_combination()
        for tf in tf_combinations:
            tf1 = [transforms.CenterCrop(IMG_SIZE)]
            tf1.extend(tf)
            tf1.append(transforms.ToTensor())
            false_aug = ImageDataset(input_file_path, DATA_ROOT_DIR, 0, transform=transforms.Compose(
                tf1
            ))
            false_train_data = ConcatDataset([false_train_data, false_aug])

        pr_true_test_data = [true_img_dataset.get(i) for i in true_test_idx]
        pr_false_test_data = [false_img_dataset.get(i) for i in false_test_idx]

        train_data = ConcatDataset([true_train_data, false_train_data])
        test_data = ConcatDataset([true_test_data, false_test_data])
        pr_test_data = ConcatDataset([pr_true_test_data, pr_false_test_data])

        train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=100, shuffle=True)
        pr_test_loader = DataLoader(pr_test_data, batch_size=100, shuffle=True)

        print('N TRUE TRAIN: {}\nN TRUE TEST: {}'.format(len(true_train_data), len(true_test_data)))
        print('N FALSE TRAIN: {}\nN FALSE TEST: {}'.format(len(false_train_data), len(false_test_data)))

        model = Net()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        test_acc = []
        test_loss = []

        for epoch in range(1, NEPOCH + 1):
            train(epoch, model, optimizer, train_loader)
            tloss, acc = test(model, test_loader)
            test_loss.append(tloss)
            test_acc.append(acc)


        acc, matrix = predict(model, pr_test_loader)
        accuracy.append(acc)
        for i in range(CLASS_NUM):
            for j in range(CLASS_NUM):
                print(matrix[i][j], end=', ')
            print('\n')

    print('mean', accuracy)
    fig, (figL, figR) = plt.subplots(ncols=2, figsize=(10,4))
    figL.plot(np.array([i for i in range(NEPOCH)]), np.array(test_loss), marker='o', linewidth=2)
    figL.set_title('test loss')
    figL.set_xlabel('epoch')
    figL.set_ylabel('loss')
    figL.set_xlim(0, NEPOCH)
    figL.grid(True)

    figR.plot(np.array([i for i in range(NEPOCH)]), np.array(test_acc), marker='o', linewidth=2)
    figR.set_title('test accuracy')
    figR.set_xlabel('epoch')
    figR.set_ylabel('accuracy')
    figR.set_xlim(0, NEPOCH)
    figR.grid(True)

    fig.show()
