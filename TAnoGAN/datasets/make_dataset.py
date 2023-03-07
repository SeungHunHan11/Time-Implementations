import json
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import torch

class create_dataset(Dataset):
    
    def __init__(self, data_dir, label_dir, train_ratio, val_ratio, window_length,
                 step_size, mode:str = 'train', scaler_type = 'minmax', split=False):
        super(create_dataset, self).__init__()


        data = pd.read_csv(data_dir)
        data['timestamp'] = pd.to_datetime(data['timestamp'])
         
        with open(label_dir) as FI:
            j_label = json.load(FI)
        
        xx, label = self.get_label(data, j_label)

        if scaler_type == 'minmax':
            getparam = self.get_minmax
            scaler = self.minmax
        else:
            getparam = self.get_meanstd
            scaler = self.standardize

        if split:

            train, val, test = self.split_dataset(x = xx, y = label, train_ratio=train_ratio, val_ratio=val_ratio)

            if train == None and val== None:
                param1, param2 = getparam(test[0])
            
            else:
                param1, param2 = getparam(train[0])

            if mode == 'train':
                self.input, self.label=self.unroll(scaler(train[0], param1, param2), train[1], window_length, step_size)
                        
            elif mode == 'validation':
                self.input, self.label=self.unroll(scaler(val[0], param1, param2), val[1], window_length, step_size)

            elif mode == 'test':
                self.input, self.label=self.unroll(scaler(test[0], param1, param2), test[1], window_length, step_size)

            else:
                raise NotImplementedError('Mode not correctly selected')
        else:
            param1, param2 = getparam(xx)
            self.input, self.label=self.unroll(scaler(xx, param1, param2), label, window_length, step_size)

    def get_label(self, df_x, j_label):
        ano_spans = j_label['realKnownCause/ambient_temperature_system_failure.csv']

        y = torch.zeros(len(df_x))

        for ano_span in ano_spans:
            ano_start = pd.to_datetime(ano_span[0])
            ano_end = pd.to_datetime(ano_span[1])
            for idx in df_x.index:
                if df_x.loc[idx, 'timestamp'] >= ano_start and df_x.loc[idx, 'timestamp'] <= ano_end:
                    y[idx] = 1.0
        
        xx = torch.from_numpy(df_x['value'].values)

        return xx, y

    def split_dataset(self, x, y, train_ratio, val_ratio):
        
        train_idx, val_idx = int(train_ratio*len(x)), int(val_ratio*len(x))
        if train_idx==0 and val_idx==0:
            test = x, y

            return None, None, test

        else:
            train = x[:train_idx], y[:train_idx]
            val = x[:train_idx+val_idx], y[:train_idx+val_idx]
            test = x[train_idx+val_idx:], y[train_idx+val_idx:]

            return train, val, test
    
    def get_minmax(self, train_data):

        minimum, maximum = train_data.min().item(), train_data.max().item()

        return minimum, maximum
    
    def get_meanstd(self, train_data):

        mean, std = (train_data*1.0).mean().item(), (train_data*1.0).std().item()

        return mean, std

    def standardize(self, data, param1, param2):
        
        return (data-param1)/(param2+1e-7)

    def minmax(self, data, param1, param2):
        
        return (data-param1)/(param2-param1)
    

    def unroll(self, xx, yy, window_size, step_size):

        seq_len = (len(xx)-window_size)//step_size +1

        window_data = torch.zeros((seq_len, window_size))
        window_label = torch.zeros((seq_len, ))

        idx = 0
        seq_idx = 0
        while(idx < xx.shape[0] - window_size + 1):
            window_data[seq_idx] = xx[idx:idx+window_size]
            window_label[seq_idx] = 1 if yy[idx:idx+window_size].sum().item() > 0 else 0
            idx += step_size
            seq_idx +=1

        return window_data, window_label
        

    def __len__(self):
        
        return self.input.shape[0]
    
    def __getitem__(self, idx):
        
        xx = self.input[idx]
        yy = self.label[idx]

        return xx.unsqueeze(-1), yy