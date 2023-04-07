from torch.utils.data import Dataset
import os
from sklearn.preprocessing import StandardScaler
import numpy as np 
import torch

class anomaly_dataset(Dataset):

    def __init__(self, data_dir: str, dataset_name:str, 
                 mode: str, scaler: str,
                 window_size: int, step_size: int, use_rawdata: bool = False):


        self.mode = mode
        self.window_size = window_size
        self.step_size = step_size

        scaler = __import__('sklearn.preprocessing', fromlist='preprocessing').__dict__[f'{scaler}']()

        if use_rawdata:
            train_data, _ = __import__('datasets').__dict__[f'{dataset_name}_dataset'](data_dir, 'train')
            test_data, label = __import__('datasets').__dict__[f'{dataset_name}_dataset'](data_dir, 'test')
        else:
            data_dir = os.path.join(data_dir, dataset_name)
            train_data = np.nan_to_num(np.load(os.path.join(data_dir,dataset_name+'_train.npy')))
            test_data = np.nan_to_num(np.load(os.path.join(data_dir,dataset_name+'_test.npy')))
            label = np.load(os.path.join(data_dir,dataset_name+'_test_label.npy'))

        self.train_data = scaler.fit_transform(train_data)
        self.test_data = scaler.transform(test_data)
        self.label = label

        if mode == 'train':
            self.data = self.train_data
            self.length = (self.train_data.shape[0] - self.window_size) // self.step_size + 1
        elif mode == 'val':
            self.data = self.test_data
            self.length = (self.test_data.shape[0] - self.window_size) // self.step_size + 1

        elif mode == 'test':
            self.data = self.test_data
            self.length = (self.test_data.shape[0] - self.window_size) // self.step_size + 1

        elif mode == 'threshold':
            self.data = self.test_data
            self.length = (self.test_data.shape[0] - self.window_size) // self.window_size + 1

        self.mode = mode

    def __len__(self):

        return self.length
    
    def __getitem__(self, idx):
         
        idx = idx * self.step_size
        
        if self.mode == 'threshold':
            data = np.float32(self.data[idx // self.step_size * self.window_size:idx // self.step_size * self.window_size + self.window_size])
            label = np.float32(self.label[idx//self.step_size * self.window_size: idx // self.step_size * self.window_size+self.window_size])
        else:
            data = np.float32(self.data[ idx : idx + self.window_size])
            label = np.float32(self.label[idx : idx + self.window_size])

        return data, label
