from glob import glob
import os
import numpy as np

def SMAP_dataset(data_dir: str, mode: str):
    
    data_dir = os.path.join(data_dir,'SMAP_MSL')
    datadir = sorted(list(set(glob(os.path.join(data_dir, mode)+'/*.npy'))))

    reserved_dir = []

    data_list = None

    for idx, dir in enumerate(datadir):
        data = np.load(dir)
        dim = data.shape[1]
        if dim == 25:
            if data_list is None:
                data_list = data
            else:
                data_list = np.append(data_list, data, axis = 0)

            reserved_dir.append(dir)
    
    label = None

    if mode != 'train':
        
        label = np.load(os.path.join(data_dir,'label.npy'))

    return data_list, label

def MSL_dataset(data_dir: str, mode: str):
    
    # data_dir = '/directory/data/Anomaly_Detection/SMAP_MSL'
    # mode = 'train'  
    datadir = sorted(list(set(glob(os.path.join(data_dir, mode)+'/*.npy'))))

    reserved_dir = []

    data_list = None

    for idx, dir in enumerate(datadir):
        data = np.load(dir)
        dim = data.shape[1]
        if dim == 55:
            if data_list is None:
                data_list = data
            else:
                data_list = np.append(data_list, data, axis = 0)

            reserved_dir.append(dir)

    return data_list

