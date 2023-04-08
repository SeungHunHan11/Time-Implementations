import torch
from torch.utils.data import DataLoader
import random
import os
import numpy as np
from log import setup_default_logging
import logging
import torch.nn as nn
from datasets import anomaly_dataset
import argparse
from inference import get_energy, evaluation
import wandb
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, f1_score
from model import Anomaly_Transformer
from train import fit
import json 
from utils import str2bool

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

def run(args):

    train_mode = str2bool(args.train_mode)
    use_scheduler = str2bool(args.use_scheduler)
    drop_pos = str2bool(args.drop_pos)
    end_to_end = str2bool(args.end_to_end)
    eval_per_epoch = str2bool(args.eval_per_epoch)
    use_wandb = str2bool(args.use_wandb)
    
    savedir = os.path.join(args.savedir, args.dataset_name, args.run_name)
    os.makedirs(savedir, exist_ok=True)    
    
    setup_default_logging(log_path=os.path.join(savedir,'log.txt'))
    torch_seed(args.seed)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    _logger.info('Device: {}'.format(device))

    train_data_set = anomaly_dataset(
                                    args.datadir,
                                    args.dataset_name, 
                                    'train', 
                                    args.scaler, 
                                    args.window_size, 
                                    args.step_size)
    
    val_data_set = anomaly_dataset(
                                args.datadir,
                                args.dataset_name, 
                                'val', 
                                args.scaler, 
                                args.window_size, 
                                args.step_size)
    
    test_data_set = anomaly_dataset(
                            args.datadir,
                            args.dataset_name, 
                            'test', 
                            args.scaler, 
                            args.window_size, 
                            args.step_size)

    threshold_data_set = anomaly_dataset(
                            args.datadir,
                            args.dataset_name, 
                            'threshold', 
                            args.scaler, 
                            args.window_size, 
                            args.step_size)


    train_loader = DataLoader(train_data_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_data_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_data_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    threshold_loader = DataLoader(threshold_data_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    sample = next(iter(train_loader))

    dim = sample[0].shape[-1]
    print(f'Train Set shape: ({len(train_data_set)}, {dim})')
    print(f'Validation Set shape: ({len(val_data_set)}, {dim})')
    print(f'Test Set shape: ({len(test_data_set)}, {dim})')
    print(f'Threshold Set shape: ({len(threshold_data_set)}, {dim})')


    model = Anomaly_Transformer(
                    window_size = args.window_size, 
                    c_in = dim, c_out = dim, 
                    d_model = args.d_model, n_head = args.n_head, 
                    num_layers = args.num_layers, d_ff = args.ffnn_dim,
                    dropout = args.dropout, 
                    activation = args.activation, norm_type = args.norm_type, 
                    pos_type = args.pos_type, emb_type = args.emb_type, 
                    device = device,
                    output_attention = True,
                    drop_pos = drop_pos
                    )

    model.to(device)
    
    # model = Anomaly_Transformer(
    #                 window_size = 100, 
    #                 c_in = 38, c_out = 38, 
    #                 d_model = 126, n_head = 6, 
    #                 num_layers = 3, d_ff = 100,
    #                 dropout = 0.1, 
    #                 activation = 'relu', norm_type = 'BatchNorm1d', 
    #                 pos_type = 'learnable', emb_type = 'Conv1d', 
    #                 device = device,
    #                 output_attention = True
    #                 )
    
    if train_mode:
        
        print('=============Train Mode==============')

        criterion = nn.MSELoss()
        optimizer =  torch.optim.Adam(model.parameters(), lr=args.lr)

        # scheduler
        if use_scheduler:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        else:
            scheduler = None
        
        if use_wandb:
            #initialize wandb
            wandb.init(name=args.run_name,group = args.dataset_name, project='Anomaly-Transformer', config=args)


        loader = threshold_loader if eval_per_epoch else None

        fit(
            model = model,
            train_loader = train_loader,
            val_loader = val_loader, 
            optimizer = optimizer, 
            criterion = criterion, 
            scheduler = scheduler, 
            device = device, 
            Lambda = args.Lambda, 
            save_dir = savedir, 
            epochs = args.epochs,
            temperature = args.temperature,
            anomaly_ratio = args.anomaly_ratio,
            threshold_loader = threshold_loader,
            use_wandb = use_wandb
            )
    
        if end_to_end:
            
            criterion_infer = nn.MSELoss(reduce=False)

            model_dir = os.path.join(savedir,  args.checkpoint_name+'.pt')

            try:
                model.load_state_dict(torch.load(model_dir))
            except:
                raise NotImplementedError()

            model.eval()

            with torch.no_grad():
                train_energy = get_energy(model = model, 
                                        criterion = criterion_infer, 
                                        data_loader = threshold_loader, 
                                        device = device,
                                        temperature = args.temperature, 
                                        anomaly_ratio = args.anomaly_ratio,
                                        train_energy = None
                                        )
                
                threshold = get_energy(model = model, 
                                        criterion = criterion_infer, 
                                        data_loader = threshold_loader, 
                                        device = device,
                                        temperature = args.temperature, 
                                        anomaly_ratio = args.anomaly_ratio,
                                        train_energy = train_energy
                                        )
                pred, ground_truth= evaluation(
                                            loader = test_loader,
                                            device = device,
                                            criterion = criterion_infer,
                                            model = model, 
                                            temperature = args.temperature,
                                            threshold = threshold
                                            )

            accuracy = accuracy_score(ground_truth, pred)
            precision, recall, f_score, support = precision_recall_fscore_support(ground_truth, pred,
                                                                                average='binary')
            f1_sc = f1_score(ground_truth, pred)

            
            eval_result = {
                        'test_Accuracy' : accuracy, 
                        'test_recall' : recall,
                        'test_fscore' : f_score,
                        'test_f1_score': f1_sc,
                        'test_precision' : precision
                        }
    

            print(
                "Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f}, F1-Score {:0.4f}".format(
                    accuracy, precision,
                    recall, f_score, f1_sc))
            
            json.dump(eval_result, open(os.path.join(savedir, "test_result.json"),'w'), indent='\t')
    
    
    else:
        print('Inference Mode')
        criterion = nn.MSELoss(reduce=False)

        model_dir = os.path.join(savedir,  args.checkpoint_name+'.pt')
        print(model_dir)

        try:
            model.load_state_dict(torch.load(model_dir))
        except:
            raise NotImplementedError()

        model.eval()

        with torch.no_grad():
            train_energy = get_energy(model = model, 
                                      criterion = criterion, 
                                      data_loader = threshold_loader, 
                                      device = device,
                                      temperature = args.temperature, 
                                      anomaly_ratio = args.anomaly_ratio,
                                      train_energy = None
                                      )
            
            threshold = get_energy(model = model, 
                                      criterion = criterion, 
                                      data_loader = threshold_loader, 
                                      device = device,
                                      temperature = args.temperature, 
                                      anomaly_ratio = args.anomaly_ratio,
                                      train_energy = train_energy
                                      )
            pred, ground_truth= evaluation(
                                        loader = test_loader,
                                        device = device,
                                        criterion = criterion_infer,
                                        model = model, 
                                        temperature = args.temperature,
                                        threshold = threshold
                                        )

        accuracy = accuracy_score(ground_truth, pred)
        precision, recall, f_score, support = precision_recall_fscore_support(ground_truth, pred,
                                                                              average='binary')
        f1_sc = f1_score(ground_truth, pred)

        
        eval_result = {
                        'test_Accuracy' : accuracy, 
                        'test_recall' : recall,
                        'test_fscore' : f_score,
                        'test_f1_score': f1_sc,
                        'test_precision' : precision
                       }
 

        print(
            "Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f}, F1-Score {:0.4f}".format(
                accuracy, precision,
                recall, f_score, f1_sc))
        
        json.dump(eval_result, open(os.path.join(savedir, "test_result.json"),'w'), indent='\t')
    
        
if __name__ =='__main__':
    
    parser = argparse.ArgumentParser(description="Classification for Computer Vision")
    parser.add_argument('--use_wandb',type=str,default=True,help='Use wandb?')

    parser.add_argument('--train_mode',type=str,default=True,help='Train Mode')
    parser.add_argument('--use_scheduler',type=str,default=True,help='Use Scheduler?')
    parser.add_argument('--eval_per_epoch',type=str,default=True,help='Evaluate per epoch?')

    parser.add_argument('--end_to_end',type=str,default=False,help='True for train and inference altogether')


    parser.add_argument('--savedir',type=str,default='./saved_models/',help='Model save directory')
    parser.add_argument('--checkpoint_name',type=str,default='best_model',help='Checkpoint name')

    parser.add_argument('--dataset_name',type=str,choices=['MSL','PSM','SMAP','SMD','SWAT'],help='Select dataset name')

    parser.add_argument('--datadir',type=str,default = './data', help='Dataset directory')

    parser.add_argument('--scaler',type=str,default = 'StandardScaler', choices = ['StandardScaler', 'MinMaxScaler'] , help='Dataset directory')
    parser.add_argument('--window_size',type=int,default = 100, help='Sliding window size')
    parser.add_argument('--step_size',type=int,default = 100, help='Sliding window step size')

    parser.add_argument('--d_model',type=int,default = 512, help='Input embedding dimension')
    parser.add_argument('--n_head',type=int,default = 6, help='Self Attention head number')

    parser.add_argument('--num_layers',type=int,default = 3, help='Encoder block number')
    parser.add_argument('--ffnn_dim',type=int,default = 512, help='Encoder FFNN dim')
    parser.add_argument('--dropout',type=float,default = 0.3, help= 'dropout ratio')

    parser.add_argument('--activation',type=str, default = 'relu', choices = ['relu', 'gelu'], help='Activation Function')
    parser.add_argument('--norm_type',type=str, default = 'BatchNorm1d', choices = ['BatchNorm1d', 'LayerNorm'], help='Normalization method')
    
    parser.add_argument('--emb_type',type=str, default = 'Linear', choices = ['Linear', 'Conv1d'], help='Embedding method')
    parser.add_argument('--pos_type',type=str, default = 'learnable', choices = ['learnable', 'encoding'], help='Positional encoding method')

    parser.add_argument('--drop_pos',type=str, default = False, help='Use positional embedding or not. True for drop pos embedding')


    parser.add_argument('--batch_size',type=int,default = 256, help='Batch size')
    parser.add_argument('--num_workers',type=int,default = 12, help='Num workers')
    parser.add_argument('--lr',type=float,default = 1e-4, help='Learning Rate')
    parser.add_argument('--epochs',type = int,default = 10, help='Number of Epochs')

    parser.add_argument('--Lambda',type=int,default = 3, help='Loss function trade-off constant')
    parser.add_argument('--temperature',type=int,default = 50, help= 'Anomaly score ratio')
    parser.add_argument('--anomaly_ratio',type=float,default = 4.00, help='Threshold hyperparameter')

    parser.add_argument('--run_name',type = str,default = 'Anomaly Transformer Default', help='Experiment Name')
    
    parser.add_argument('--seed', type = int, default = 1998, help = 'Seed Number')
    
    args = parser.parse_args()

    run(args)
