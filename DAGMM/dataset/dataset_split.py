import torch

def split(window,split_ratio: list = [0.6,0.1,0.3]):

    seq_len=window.shape[0]

    train_len=int(seq_len*split_ratio[0])
    val_len=int(seq_len*split_ratio[1])

    train_window=window[:train_len,:]
    val_window=window[train_len:train_len+val_len,:]

    test_window=window[train_len+val_len:,:]

    minimum=torch.min(train_window,dim=0).values
    maximum=torch.max(train_window,dim=0).values

    denominator=(maximum-minimum)+0.1

    train_norm=torch.subtract(train_window,minimum)/denominator
    val_norm=torch.subtract(val_window,minimum)/denominator
    test_norm=torch.subtract(test_window,minimum)/denominator


    return train_norm, val_norm, test_norm