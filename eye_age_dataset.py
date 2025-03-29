import pandas as pd
import torch
import os
import numpy as np
from skimage import transform
from PIL import Image

from torch.utils.data import Dataset, DataLoader

class EyeAgeDataset(Dataset):
    # def __init__(self, csv_file='../data.csv', root_dir='flipped_good_quality', transform=None, fold=0, is_train=True):
    def __init__(self, csv_file='/home/ttn/Development/eye-age/aireadi_data.csv', root_dir='/home/ttn/Development/eye-age/AutoMorph/images', transform=None, fold=0, is_train=True):
    
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data = pd.read_csv(csv_file)
        self.data = self.data[(self.data.anatomic_region.str.contains('Macula')) & (~self.data.anatomic_region.str.contains('Optic')) & (self.data.imaging.str.contains('Color Photography'))]
        self.data['label'] = (self.data['moca'] <= 25).astype(int)
        self.train_set = self.data[self.data['fold'] != fold]
        if is_train == 'test':
            self.data = self.data[self.data['fold'] == fold]
            # self.data = pd.read_csv('/home/ttn/Development/eye-age/aireadi_data.csv')
            print(f'test data shape: {self.data.shape}')
            # root_dir = '/home/ttn/Development/eye-age/AutoMorph/Results/M1/Good_quality'
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
        target = self.data['label'].iloc[idx]
        sample = image
        path = self.data['path'].iloc[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample, target, path