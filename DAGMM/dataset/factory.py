from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import pandas as pd
import torch
from tqdm import tqdm
from dataset.dataset_split import split
import os

class TSDataset(Dataset):
    def __init__(self,
                 df: pd.DataFrame,
                 label: pd.DataFrame,
                 mode: str = 'train',
                 input_dim: int = 11,
                 windows_size: int = 1, slide_size: int = 1,
                 split_ratio: list = [0.6, 0.1, 0.3]):

        self.window_size = windows_size
        self.slide_size = slide_size

        if os.path.exists('./dataset/train.pt')*os.path.exists('./dataset/val.pt')*os.path.exists('./dataset/test.pt'):
            
            train_load = torch.load('./dataset/train.pt')
            val_load = torch.load('./dataset/val.pt')
            test_load = torch.load('./dataset/test.pt')

            train, train_label = train_load['data'], train_load['label']
            val, val_label = val_load['data'], val_load['label']
            test, test_label = test_load['data'], test_load['label']

        else:
            seq_len = (len(df)-windows_size)//slide_size+1
            sliding_window = torch.zeros((seq_len, windows_size, input_dim))
            label_tensor = torch.zeros((seq_len, ))

            for i in tqdm(range(seq_len)):
                sliding_window[i, :, :] = torch.tensor(df[i:i+windows_size].values)
                label_tensor[i] = torch.tensor(label.loc[i].values,dtype=torch.float16)

            train, val, test = split(sliding_window, split_ratio) 


            train_len, val_len=int(seq_len*split_ratio[0]), int(seq_len*split_ratio[1])

            train_label=label_tensor[:train_len]

            train = train[train_label==0] # Only use Normal data for train

            train_label=train_label[train_label==0]
            
            # Note: Sliding Window not really implemented as DAGMM takes single entity as an input. 
            # Thus, sliding window at current state is just for future implementation.

            train= train.reshape(train.size(0),-1)
            val = val.reshape(val.size(0),-1)
            test = test.reshape(test.size(0),-1)

            val_label=label_tensor[train_len:train_len+val_len]
            test_label=label_tensor[train_len+val_len:]

            torch.save({'data':train,'label':train_label}, './dataset/train.pt')
            torch.save({'data':val,'label':val_label}, './dataset/val.pt')
            torch.save({'data':test,'label':test_label}, './dataset/test.pt')

        self.sliding_window = train
        self.label = train_label

        if mode == 'val':
            self.sliding_window = val
            self.label = val_label
        elif mode == 'test':
            self.sliding_window = test
            self.label = test_label

    def __len__(self):
        return self.label.shape[0]

    def __getitem__(self, idx):
        
        return self.sliding_window[idx], self.label[idx]


def TSLoader(dataset, batch_size: int = 8, shuffle = False, num_workers: int = 12):

    return DataLoader(dataset, batch_size, shuffle = shuffle , num_workers=num_workers)