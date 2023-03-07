from train import eval, fit
from models import Gen, Dis
from datasets import create_dataset
from torch.utils.data import DataLoader
import wandb
import os
import torch
import random
import numpy as np
import logging
from log import setup_default_logging
import warnings
import argparse
import yaml
from optimize import optimize
import json
import pdb
_logger = logging.getLogger('train')

def torch_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU 
    # CUDA randomness
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    np.random.seed(random_seed)
    random.seed(random_seed)
    os.environ['PYTHONHASHSEED'] = str(random_seed)

def run(cfg):

    save_dir = os.path.join(cfg['save_dir'],cfg['EXP_NAME'])
    os.makedirs(save_dir, exist_ok = True)

    setup_default_logging(log_path=os.path.join(save_dir,'log.txt'))

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    _logger.info('Device: {}'.format(device))

    torch_seed(cfg['SEED'])

    gen_model = Gen(cfg['DATASET']['in_dim'], cfg['DATASET']['window_size'])
    gen_model.to(device)
    dis_model = Dis(cfg['DATASET']['in_dim'], cfg['DATASET']['hidden_dim'])
    dis_model.to(device)
    

    if cfg['MODE']['Train_model']:
        
        train_set = create_dataset(cfg['DATASET']['data_dir'], 
                                    cfg['DATASET']['label_dir'],
                                    cfg['DATASET']['TRAIN_RATIO'],
                                    cfg['DATASET']['VAL_RATIO'],
                                    cfg['DATASET']['window_size'],
                                    cfg['DATASET']['step_size'],
                                    mode='train',
                                    scaler_type = cfg['DATASET']['scaler'],
                                    split = cfg['DATASET']['split']
                                    )
        train_loader = DataLoader(train_set, cfg['DATASET']['batch_size'], shuffle = True, num_workers= 12)
        
        val_loader = None

        if cfg['DATASET']['split']:
            val_set = create_dataset(cfg['DATASET']['data_dir'], 
                                    cfg['DATASET']['label_dir'],
                                        cfg['DATASET']['TRAIN_RATIO'],
                                        cfg['DATASET']['VAL_RATIO'],
                                        cfg['DATASET']['window_size'],
                                        cfg['DATASET']['step_size'],
                                        mode='validation',
                                        scaler_type = cfg['DATASET']['scaler'],
                                        split = cfg['DATASET']['split']
                                        )
        
            val_loader = DataLoader(val_set, cfg['DATASET']['batch_size'], shuffle = False, num_workers= 12)

        criterion = torch.nn.BCELoss()
        optimizer_g = torch.optim.Adam(params=gen_model.parameters(), lr = cfg['TRAIN']['lr'])
        optimizer_d = torch.optim.Adam(params=dis_model.parameters(), lr = cfg['TRAIN']['lr'])
        
        if cfg['TRAIN']['use_wandb']:
            wandb.init(name=cfg['EXP_NAME'], project='TAnoGAN', config=cfg)

        fit(
            gen = gen_model,
            dis = dis_model,
            train_loader = train_loader,
            val_loader = val_loader,
            criterion = criterion,
            optimizer_g = optimizer_g,
            optimizer_d = optimizer_d,
            epochs = cfg['TRAIN']['epochs'],
            save_dir = save_dir,
            log_interval = cfg['TRAIN']['log_interval'],
            eval_interval = cfg['TRAIN']['eval_interval'],
            device = device,
            use_wandb = cfg['TRAIN']['use_wandb'],
            split = cfg['DATASET']['split']
        )

        _logger.info('MODEL TRAINING COMPLETED. \n BEST RESULT SAVED')
        
        if cfg['DATASET']['split']:
            # Log Test Results
            gen_dir = os.path.join(save_dir,cfg['Saved_models']['gen_dir'])
            dis_dir = os.path.join(save_dir,cfg['Saved_models']['dis_dir'])
            gen_model.load_state_dict(torch.load(gen_dir))
            dis_model.load_state_dict(torch.load(dis_dir))
            gen_model.to(device), dis_model.to(device)

            test_set = create_dataset(cfg['DATASET']['data_dir'], 
                                    cfg['DATASET']['label_dir'],
                                        cfg['DATASET']['TRAIN_RATIO'],
                                        cfg['DATASET']['VAL_RATIO'],
                                        cfg['DATASET']['window_size'],
                                        cfg['DATASET']['step_size'],
                                        mode = 'test',
                                        scaler_type = cfg['DATASET']['scaler'],
                                        split = cfg['DATASET']['split']
                                        )
            
            test_loader = DataLoader(test_set, cfg['DATASET']['batch_size'], shuffle = False, num_workers=12)


            test_metrics=eval(gen_model, dis_model,val_loader=test_loader,criterion=criterion,
                                device=device, log_interval=cfg['TRAIN']['eval_interval'])

            json.dump(test_metrics, open(os.path.join(save_dir,'test_result.json'),'w'), indent='\t')
            

    if cfg['MODE']['Optimize']:

        test_set = create_dataset(cfg['DATASET']['data_dir'], 
                    cfg['DATASET']['label_dir'],
                    0,
                    0,
                    cfg['DATASET']['window_size'],
                    cfg['DATASET']['window_size'],
                    mode='test',
                    scaler_type = cfg['DATASET']['scaler'],
                    split = cfg['DATASET']['split']
                    )
        
        test_loader = DataLoader(test_set, 1, shuffle = False, num_workers=12)
        print(len(test_loader)*1)
        if cfg['Saved_models']['lastascheckpoint']:
            gen_dir = os.path.join(save_dir,cfg['Saved_models']['last_gen_dir'])
            dis_dir = os.path.join(save_dir,cfg['Saved_models']['last_dis_dir'])
            df_dir = os.path.join(save_dir,cfg['DATASET']['scaler']+'_laststep_Test_AD_results.csv')
        else:
            gen_dir = os.path.join(save_dir,cfg['Saved_models']['gen_dir'])
            dis_dir = os.path.join(save_dir,cfg['Saved_models']['dis_dir'])
            df_dir = os.path.join(save_dir,cfg['DATASET']['scaler']+'_beststep_Test_AD_results.csv')

        if os.path.exists(gen_dir) and os.path.exists(dis_dir):
            gen_model.load_state_dict(torch.load(gen_dir))
            dis_model.load_state_dict(torch.load(dis_dir))
            gen_model.to(device), dis_model.to(device)
        else:
            raise NotImplementedError('There are no trained Discriminator or Generator')

        optimize(save_dir=df_dir,dis=dis_model,gen = gen_model, test_loader = test_loader,
                    window_size= cfg['DATASET']['window_size'],
                    in_dim=cfg['DATASET']['in_dim'],
                    Lambda=cfg['Optimize']['lambda'],
                    iterations= cfg['Optimize']['iterations'],
                    device = device, log_interval = int(cfg['Optimize']['iterations']/5),
                    )

if __name__ == '__main__':

    warnings.filterwarnings('ignore')
    parser = argparse.ArgumentParser(description='TAnoGAN Train Gen and Dis')
    parser.add_argument('--yaml_config', type=str, default=None, help='exp config file')    

    args = parser.parse_args()
    cfg = yaml.load(open(args.yaml_config,'r'), Loader=yaml.FullLoader)

    run(cfg)

    # train=create_dataset('/workspace/NAB/data/realKnownCause/nyc_taxi.csv', '/workspace/NAB/labels/combined_windows.json',
    #             0.6, 0.1,60,1 )

    # loader = DataLoader(train,32)

    # sample = next(iter(loader))

    # device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


    # gen_model = Gen(1, 60)
    # gen_model.to(device)

    # h_0, c_0 = torch.zeros(1, 32, 32).to(device), torch.zeros(1, 32, 32).to(device)
    # xx=sample[0].to(device)

    # gen_model(xx, h_0, c_0)

    # dis_model = Dis(1, 100)
    # dis_model.to(device)

    # h_0, c_0 = torch.zeros(1, 32, 100).to(device), torch.zeros(1, 32, 100).to(device)
    # xx.shape

    # out = dis_model(xx, h_0,c_0)

    # out.shape
    # pred = torch.round(out).detach().cpu()

    # pred.shape
    # label.shape

    # label.detach().cpu().eq(pred).sum().item()

    # out.mean(dim=2).item()

    # criterion = torch.nn.BCELoss()
    # label = torch.ones((32, 60, 1)).to(device)

    # criterion(out,label)

