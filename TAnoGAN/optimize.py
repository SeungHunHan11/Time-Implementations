import torch
from torch.autograd import Variable
import torch.nn.init as init
import pandas as pd
import os
import logging

_logger = logging.getLogger('Optimization')

def Anomaly_score(latent, fake, latent_interm, fake_interm, Lambda=0.1):
    residual_loss = torch.sum(torch.abs(latent-fake)) # Residual Loss
    
    discrimination_loss = torch.sum(torch.abs(latent_interm-fake_interm)) # Discrimination loss
    
    total_loss = (1-Lambda)*residual_loss + Lambda*discrimination_loss
    return total_loss

def optimize(save_dir, dis, gen, test_loader, window_size, 
             in_dim, Lambda, iterations, device, log_interval):

    anomaly_score = []
    y_label = []
    actual_obs = []
    test_anomaly_score = pd.DataFrame()

    dis.eval()
    gen.eval()

    torch.backends.cudnn.enabled = False
    
    for idx, (xx,yy) in enumerate(test_loader):
        import pdb

        y_label.append(yy.detach().item())
        actual_obs.append(xx.flatten()[window_size-1].detach().item())

        xx = xx.to(device)
        batch_size = xx.size(0)

        h_0, c_0 = torch.zeros(1, batch_size, dis.hidden_size).to(device), torch.zeros(1, batch_size, dis.hidden_size).to(device)
        h_g_0, c_g_0 = torch.zeros(1, batch_size, 32).to(device), torch.zeros(1, batch_size, 32).to(device)

        z = Variable(init.normal(torch.zeros(batch_size, window_size, in_dim, device=device),mean = 0, std = 0.1),
                    requires_grad = True)

        z_optimizer =  torch.optim.Adam([z],lr = 0.01)
        
        _logger.info(f'\n Batch: {idx+1}/{len(test_loader)}')

        for iter in range(iterations):
            fake, _ = gen(z, h_g_0, c_g_0)
            _, x_feature = dis(xx, h_0, c_0) 
            _, G_z_feature = dis(fake, h_0, c_0) 

            loss = Anomaly_score(Variable(xx), fake, x_feature, G_z_feature, Lambda = Lambda)
            loss.backward()
            z_optimizer.step()
            
        _logger.info('Batch [{}/{}]: Anomaly Score: {:.3f} label: {}'.format(idx+1, len(test_loader), loss.item(),yy.detach().item()))
                                                                                        
        anomaly_score.append(loss.item())

    test_anomaly_score['actual_obs'] = actual_obs
    test_anomaly_score['score'] = anomaly_score
    test_anomaly_score['true_label'] = y_label

    test_anomaly_score.to_csv(save_dir,index=False)