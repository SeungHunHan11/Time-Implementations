import torch
from tqdm import tqdm
from dataset.dataset_split import split
import yaml
import argparse
import pandas as pd
import os

def save(
        df: pd.DataFrame, 
        label: pd.DataFrame, 
        save_dir: str,
        input_dim: int = 11,
        windows_size: int = 1,
        slide_size: int = 1,
        split_ratio: list = [0.6, 0.1, 0.3]
        ):

    seq_len = (len(df)-windows_size)//slide_size+1
    sliding_window = torch.zeros((seq_len, windows_size, 11))
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

    torch.save({'data':train,'label':train_label}, os.path.join(save_dir,'train.pt'))
    torch.save({'data':val,'label':val_label}, os.path.join(save_dir,'val.pt'))
    torch.save({'data':test,'label':test_label}, os.path.join(save_dir,'test.pt'))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml_config', type=str, default=None, help='exp config file')
    args = parser.parse_args()

    cfg = yaml.load(open(args.yaml_config,'r'), Loader=yaml.FullLoader)

    df = pd.read_csv(cfg['input_dir'])
    label = pd.read_csv(cfg['label_dir'])

    save(df, label,
        save_dir = cfg['save_dir'],
        input_dim = cfg['input_dim'],
        windows_size = cfg['window_size'],
        slide_size = cfg['slide_size'],
        split_ratio = cfg['split_ratio']
        )