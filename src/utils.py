from dataclasses import replace
import os
from time import time
from copy import deepcopy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2 as cv

import torch
from torch.utils.data import Dataset
from torchvision.utils import make_grid

# class ImageFolderDataset(Dataset):
#     '''
#     Dataset folder structure should be like: root/subdatasetes/classes/images
#     '''
#     def __init__(self, root_directory):
#         super().__init__()
#         self.root_directory = root_directory
#         self.subdatasets = os.listdir(root_directory)
#         self.classes = set([]).union(*[os.listdir(os.path.join(root_directory, subdataset)) for subdataset in self.subdatasets])
#         self.encoder = LabelEncoder()
#         self.encoder.fit(self.classes)
#         self.data = pd.DataFrame(columns=['images', 'labels'])

class MyDataset(Dataset):
    def __init__(self, image_data, labels, index,  transform=None):
        super().__init__()
        self.data = image_data
        self.labels = labels
        self.index = index
        self.transform = transform

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        image = self.data[self.index[idx]]
        label = self.labels[self.index[idx]]
        if self.transform:
            image = self.transform(image)
        return image, label

    def shuffle(self):
        np.random.shuffle(self.index)

class TotalDataset:
    def __init__(self, dataset_dir, transforms=None):
        self.data, self.labels = folderDatasetToArray(dataset_dir, transforms)
        self.total = len(self.labels)
        self.reset()

    def reset(self):
        self.unselected_normal_index = np.where(np.array(self.labels) == 1)[0]
        self.unselected_abnormal_index = np.where(np.array(self.labels) == 0)[0]

    def selectData(self, total, ratio, augmentation=1):
        normal = int(total * ratio)
        abnormal = total - normal
        if normal > self.unselected_normal_index.shape[0] or abnormal > self.unselected_abnormal_index.shape[0]:
            raise Exception('Not enough data')

        normal_index = np.random.choice(self.unselected_normal_index, normal, replace=False)
        abnormal_index = np.random.choice(self.unselected_abnormal_index, abnormal, replace=False)
        self.unselected_normal_index = np.setdiff1d(self.unselected_normal_index, normal_index)
        self.unselected_abnormal_index = np.setdiff1d(self.unselected_abnormal_index, abnormal_index)

        if augmentation > 1:
            abnormal_index = np.repeat(abnormal_index, augmentation)
        
        return MyDataset(self.data, self.labels, np.r_[normal_index, abnormal_index])    

def folderDatasetToArray(dataset_dir, transforms=None):
    data = []
    labels = []
    classes = os.listdir(dataset_dir)
    print(f'\rLoading dataset "{os.path.split(dataset_dir)[-1]}"')
    start = time()
    for class_index, class_name in enumerate(classes):
        images = os.listdir(os.path.join(dataset_dir, class_name))
        image_count = len(images)
        for i, image_name in enumerate(images):
            image = cv.imread(os.path.join(dataset_dir, class_name, image_name))
            if transforms is not None:
                image = transforms(image)
            data.append(image)
            labels.append(np.int64(class_index))
            progress = (i + 1) * 30 // image_count
            print(f'\rLoading class: {class_name} {"*" * progress}{"-" * (30 - progress)} {i + 1}/{image_count} {image_name}', end='\t')
        print(f'\rLoading class: {class_name} {"*" * progress}{"-" * (30 - progress)} {i + 1}/{image_count} Finished! Time used: {time() - start:.3f}s')
    return data, labels

def splitTrainValTest(image_data, labels, val_ratio, test_ratio, train_transforms=None):
    total = len(image_data)
    index = np.arange(total)
    np.random.shuffle(index)
    val_size = int(total * val_ratio)
    test_size = int(total * test_ratio)

    val_index = index[:val_size]
    test_index = index[val_size:val_size + test_size]
    train_index = index[val_size + test_size:]

    val_dataset = MyDataset(image_data, labels, val_index)
    test_dataset = MyDataset(image_data, labels, test_index)
    train_dataset = MyDataset(image_data, labels, train_index, train_transforms)
    return train_dataset, val_dataset, test_dataset

def showDataloaderImage(dataloader, title=None):
    data, labels = next(iter(dataloader))
    display = make_grid(data)
    display = display.numpy().transpose((1, 2, 0))
    display *= [0.229, 0.224, 0.225]
    display += [0.485, 0.456, 0.406]
    display = np.clip(display, 0, 1)
    plt.imshow(display)

