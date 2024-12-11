import pandas as pd
import torch
import os
import numpy as np
from skimage import transform
from PIL import Image

from torch.utils.data import Dataset, DataLoader


class EyeAgeDataset(Dataset):
    def __init__(self, csv_file='../data.csv', root_dir='Good_quality', transform=None, fold=0, is_train=True):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data = pd.read_csv(csv_file)
        self.train_set = self.data[(self.data['fold'] != fold) & (self.data['val_fold'] != 0)]
        self.train_mean_age = self.train_set['Age'].mean()
        self.train_std_age = self.train_set['Age'].std()
        if is_train == 'test':
            self.data = self.data[self.data['fold'] == fold]
            # self.data = pd.read_csv('/home/ttn/Development/primates/data.csv')
            # root_dir = '/home/ttn/Development/primates/AutoMorph/Results/M1/Good_quality'
        else:
            self.data = self.data[self.data['fold'] != fold]
            if is_train == 'train':
                self.data = self.data[self.data['val_fold'] != 0]
            elif is_train == 'val':
                self.data = self.data[self.data['val_fold'] == 0]
        
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.data.path.iloc[idx])
        with open(img_name, 'rb') as f:
            image = Image.open(f).convert('RGB')
        age = self.data['Age'].iloc[idx]
        sample = image
        target = age
        path = self.data['path'].iloc[idx]
        # target = (age - self.train_mean_age) / self.train_std_age

        if self.transform:
            sample = self.transform(sample)

        return sample, target, path