import torch
from model import DAGMM
from dataset import TSDataset, TSLoader
import pandas as pd
from utils import loss_func, get_gmm_param, sample_energy, torch_seed
import os
import argparse
import yaml
from tqdm import tqdm

def run(cfg):
    
    torch_seed(cfg['TRAIN']['SEED'])

    model = DAGMM(cfg['MODEL']['input_dim'],cfg['MODEL']['hidden_dim'],
            cfg['MODEL']['latent_dim'],cfg['MODEL']['dropout'],
            cfg['MODEL']['n_gmm'])

    model_dir = cfg['MODEL']['best_dir']
    model.load_state_dict(torch.load(model_dir))

    df = pd.read_csv(cfg['DATASET']['input_dir'])
    label = pd.read_csv(cfg['DATASET']['label_dir'])

    trainset = TSDataset(df, label, mode = 'train', 
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

    test_loader = TSLoader(testset, batch_size = cfg['DATASET']['batch_size'],
                            shuffle = False, num_workers = cfg['DATASET']['num_workers'])

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    criterion = loss_func

    lambda1 = cfg['TRAIN']['lambda1']
    lambda2 = cfg['TRAIN']['lambda2']

    energy_list = []
    label_list = []
    sum_prob, sum_mean, sum_cov = 0,0,0
    data_size = 0

    model.to(device)
    model.eval()

    with torch.no_grad():
        for idx, (xx, label) in enumerate(tqdm(train_loader)):
            xx = xx.to(device)
            z_c, reconstructed, z, gamma = model(xx)
            loss, recon_error, e, p = criterion(z_c.cpu(), reconstructed.cpu(), gamma.cpu(), z.cpu(), lambda1, lambda2)
            m_prob, m_mean, m_cov = get_gmm_param(gamma, z)

            sum_prob += m_prob
            sum_mean += m_mean * m_prob.unsqueeze(1)
            sum_cov += m_cov * m_prob.unsqueeze(1).unsqueeze(1)
            
            data_size += xx.shape[0]

        train_prob = sum_prob / data_size
        train_mean = sum_mean / sum_prob.unsqueeze(1)
        train_cov = sum_cov / m_prob.unsqueeze(1).unsqueeze(1)

        for idx, (xx, label) in enumerate(tqdm(test_loader)):
            xx = xx.squeeze(1).to(device)
            _, _, z, gamma = model(xx)


            for i in range(z.shape[0]):

                zi = z[i].unsqueeze(1)            
                sample_energy_value = sample_energy(train_prob.cpu(), 
                                                    train_mean.cpu(), 
                                                    train_cov.cpu(), 
                                                    zi.cpu(), 
                                                    gamma.shape[1],
                                                    gamma.shape[0])
                
                se = sample_energy_value.detach().item()

                energy_list.append(se)
                label_list.append(int(label[i].item()))

    anomaly_score = pd.DataFrame()

    anomaly_score['anomaly_score'] = energy_list
    anomaly_score['label'] = label_list

    anomaly_score.to_csv(os.path.join(cfg['SAVE']['save_dir'], 'inference.csv'),index=False)

if __name__=='__main__':
    
    parser = argparse.ArgumentParser(description='DAGMM')
    parser.add_argument('--yaml_config', type=str, default=None, help='exp config file')    
    args = parser.parse_args()

    cfg = yaml.load(open(args.yaml_config,'r'), Loader=yaml.FullLoader)
    
    run(cfg)