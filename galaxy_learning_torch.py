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


class FitsImageDataset(Dataset):

    def __init__(self, csv_file_path, root_dir, label, transform=None):
        tmp_dataframe = pd.read_csv(csv_file_path, header=None)
        self.image_dataframe = tmp_dataframe[tmp_dataframe[LABEL_IDX] == label]
        if label == 1:
            self.image_dataframe = self.image_dataframe#.sample(n=200)
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


    def __getitem__(self, idx):
        img_id = self.image_dataframe.iat[idx, 0]
        img_name = self.image_dataframe.iat[idx, 1]
        img_names = self.image_dataframe.iloc[idx, IMG_IDX:IMG_IDX+IMG_CHANNEL]
        img_names = [ path for path in img_names ]
        image = self.load_image(img_names)
        label = self.image_dataframe.iat[idx, LABEL_IDX]

        if self.transform:
            image = self.transform(image)

        return img_id, img_name, img_names, image, label


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

        image /= 255

        return img_id, img_name, img_names, image, label


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
    tr.append([transforms.RandomRotation((44, 45))])
    tr.append([transforms.RandomRotation((89, 90))])
    tr.append([transforms.RandomRotation((134, 135))])
    tr.append([transforms.RandomRotation((179, 180))])
    tr.append([transforms.RandomRotation((224, 225))])
    tr.append([transforms.RandomRotation((269, 270))])
    tr.append([transforms.RandomRotation((314, 315))])
    tr.append([transforms.RandomHorizontalFlip(1)])
    #tr.append([transforms.RandomRotation((44, 45)), transforms.RandomHorizontalFlip(1)])
    #tr.append([transforms.RandomRotation((89, 90)), transforms.RandomHorizontalFlip(1)])
    #tr.append([transforms.RandomRotation((134, 135)), transforms.RandomHorizontalFlip(1)])
    #tr.append([transforms.RandomRotation((179, 180)), transforms.RandomHorizontalFlip(1)])
    #tr.append([transforms.RandomRotation((224, 225)), transforms.RandomHorizontalFlip(1)])
    #tr.append([transforms.RandomRotation((269, 270)), transforms.RandomHorizontalFlip(1)])
    #tr.append([transforms.RandomRotation((314, 315)), transforms.RandomHorizontalFlip(1)])
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
        return F.softmax(x, dim=1)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=2):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(IMG_CHANNEL, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(2, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x



criterion = nn.CrossEntropyLoss()


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

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), accuracy))
    print(conf_matrix)

    return test_loss, accuracy


def write_result(img_ids, img_name, img_names, labels, probs, result_file):
    print('writing result...')
    #print(img_ids, img_name, img_names, labels, probs)
    with open(os.path.join(DATA_ROOT_DIR, result_file), 'a') as f:
        writer = csv.writer(f, lineterminator='\n')
        for img_id, img_name, src_imgs, label, prob in\
                zip(img_ids, img_name, img_names, labels, probs):

            result = [str(img_id.item()), img_name, src_imgs]
            result.append(label.item())
            for data in prob:
                result.append(data.item())
            writer.writerow(result)


def predict(model, test_loader, result_file):
    model.eval()
    correct = 0
    conf_matrix = [ [ 0 for j in range(CLASS_NUM) ] for i in range(CLASS_NUM)]
    for (img_ids, img_name, img_names, image, label) in test_loader:
        with torch.no_grad():
            if GPU:
                image, label = image.cuda(), label.cuda()
            image, label = Variable(image.float()), Variable(label)
            output = model(image)
            print(output.data[0], 'a')
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            flatten_label = label.data.view_as(pred)
            correct += pred.eq(flatten_label).long().cpu().sum()
            for i in range(CLASS_NUM):
                for j in range(CLASS_NUM):
                    conf_matrix[i][j] += pred[flatten_label==i].eq(j).long().cpu().sum().item()
            write_result(img_ids, img_name, img_names, label, output.data, result_file)

    accuracy = 100. * ( correct / len(test_loader.dataset) )
    print(conf_matrix)

    return accuracy, conf_matrix


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

            model = ResNet(BasicBlock, [3, 7, 6, 4])
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
