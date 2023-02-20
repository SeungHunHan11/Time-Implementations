from utils import torch_seed
from log import setup_default_logging
import logging
from model import DAGMM
import os
import torch
import numpy as np
from dataset import TSDataset, TSLoader
import pandas as pd
from train import fit
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
import argparse
import yaml

from utils import loss_func
import wandb

_logger = logging.getLogger('train')

def run(cfg):

    save_dir = os.path.join(cfg['save_dir'], cfg['exp_name'])
    os.makedirs(save_dir, exist_ok=True)
    
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    df = pd.read_csv(cfg['DATASET']['input_dir'])
    label = pd.read_csv(cfg['DATASET']['label_dir'])

    setup_default_logging(log_path=os.path.join(save_dir,'log.txt'))

    torch_seed(cfg['TRAIN']['SEED'])

    model = DAGMM(cfg['MODEL']['input_dim'],cfg['MODEL']['hidden_dim'],
                cfg['MODEL']['latent_dim'],cfg['MODEL']['dropout'],
                cfg['MODEL']['n_gmm'])
    
    model.to(device)

    _logger.info('# of params: {}'.format(np.sum([p.numel() for p in model.parameters()])))

    trainset = TSDataset(df, label, mode = 'train', 
                                input_dim = cfg['MODEL']['input_dim'],
                                windows_size = cfg['DATASET']['window_size'],
                                slide_size = cfg['DATASET']['slide_size'],
                                split_ratio = cfg['DATASET']['split_ratio'])

    valset = TSDataset(df, label, mode = 'val', 
                                input_dim = cfg['MODEL']['input_dim'],
                                windows_size = cfg['DATASET']['window_size'],
                                slide_size = cfg['DATASET']['slide_size'],
                                split_ratio = cfg['DATASET']['split_ratio'])

    testset = TSDataset(df, label, mode = 'test',
                                input_dim = cfg['MODEL']['input_dim'],
                                windows_size = cfg['DATASET']['window_size'],
                                slide_size = cfg['DATASET']['slide_size'],
                                split_ratio = cfg['DATASET']['split_ratio'])



    train_loader = TSLoader(trainset, batch_size = cfg['DATASET']['batch_size'],
                            shuffle = True, num_workers = cfg['DATASET']['num_workers'])

    val_loader = TSLoader(valset, batch_size = cfg['DATASET']['batch_size'],
                            shuffle = False, num_workers = cfg['DATASET']['num_workers'])

    test_loader = TSLoader(testset, batch_size = cfg['DATASET']['batch_size'],
                            shuffle = False, num_workers = cfg['DATASET']['num_workers'])
        
    print('Train Loader length: {} \n Validaiton Loader length: {} \n Test Loader length: {}'.format(len(train_loader),len(val_loader),len(test_loader)))

    optimizer = Adam(model.parameters(),cfg['TRAIN']['lr'], amsgrad=True)
    scheduler = MultiStepLR(optimizer, [5, 8], 0.1)

    criterion = loss_func

    if cfg['TRAIN']['use_wandb']:
        wandb.init(name=cfg['exp_name'], project='DAGMM', config=cfg)


    fit(
        model = model,
        train_loader = train_loader, 
        val_loader = val_loader,
        criterion = criterion,
        optimizer = optimizer,
        scheduler = scheduler,
        device = device, 
        save_dir = save_dir,
        use_wandb = cfg['TRAIN']['use_wandb'],
        lambda1 = cfg['TRAIN']['lambda1'],
        lambda2 = cfg['TRAIN']['lambda2'],
        epochs = cfg['TRAIN']['epochs'], 
        log_interval = cfg['LOG']['log_interval'], 
        eval_interval = cfg['LOG']['eval_interval']
    )

if __name__=='__main__':
    
    parser = argparse.ArgumentParser(description='DAGMM')
    parser.add_argument('--yaml_config', type=str, default=None, help='exp config file')    
    args = parser.parse_args()

    cfg = yaml.load(open(args.yaml_config,'r'), Loader=yaml.FullLoader)

    run(cfg)