import matplotlib.pyplot as plt
import torch
import math
import numpy as np
import os
import random

def reconstruct_error(x, x_hat):   
    e = torch.tensor(0.0)
    for i in range(x.shape[0]):
        e += torch.dist(x[i], x_hat[i])
    return e / x.shape[0]

def get_gmm_param(gamma, z):

    N = gamma.shape[0]
    ceta = torch.sum(gamma, dim=0) / N  #shape: [n_gmm]
    
    mean = torch.sum(gamma.unsqueeze(-1) * z.unsqueeze(1), dim=0)
    mean = mean / torch.sum(gamma, dim=0).unsqueeze(-1)  #shape: [n_gmm, z_dim]
        

    z_mean = (z.unsqueeze(1)- mean.unsqueeze(0))
    cov = torch.sum(gamma.unsqueeze(-1).unsqueeze(-1) * z_mean.unsqueeze(-1) * z_mean.unsqueeze(-2), dim = 0) / torch.sum(gamma, dim=0).unsqueeze(-1).unsqueeze(-1)
    
    return ceta, mean, cov

def sample_energy(ceta, mean, cov, zi, n_gmm, bs):
    # print('calculate sample energy')
    e = torch.tensor(0.0)
    cov_eps = torch.eye(mean.shape[1]) * (1e-3) # original constant: 1e-12
#         cov_eps = cov_eps.to(device)
    for k in range(n_gmm):
        miu_k = mean[k].unsqueeze(1)
        d_k = zi - miu_k

        inv_cov = torch.inverse(cov[k] + cov_eps)
        
        e_k = torch.exp(-0.5 * torch.chain_matmul(torch.t(d_k), inv_cov, d_k))
        e_k = e_k / torch.sqrt(torch.abs(torch.det(2 * math.pi * cov[k])))
        e_k = e_k * ceta[k]
        e += e_k.squeeze()
        
    return -torch.log(e)

def loss_func(x, dec, gamma, z, lambda1, lambda2):
    bs,n_gmm = gamma.shape[0],gamma.shape[1]
    
    #1
    recon_error = reconstruct_error(x, dec)
    
    #2
    ceta, mean, cov = get_gmm_param(gamma, z)
    
    #3
    e = torch.tensor(0.0)
    for i in range(z.shape[0]):
        zi = z[i].unsqueeze(1)
        ei = sample_energy(ceta, mean, cov, zi,n_gmm,bs)
        e += ei
    
    p = torch.tensor(0.0)
    for k in range(n_gmm):
        cov_k = cov[k]
        p_k = torch.sum(1 / torch.diagonal(cov_k, 0))
        p += p_k


    loss = recon_error + (lambda1 / z.shape[0]) * e   + lambda2 * p
    
    return loss, recon_error, e/z.shape[0], p

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

def early_stop():
    pass

def plot_loss_moment(losses,n_gmm):

    _, ax = plt.subplots(figsize=(10,5), dpi=80)
    ax.plot(losses, 'blue', label='train', linewidth=1)
    ax.set_title('Loss change in training')
    ax.set_ylabel('Loss')
    ax.set_xlabel('Iteration')
    ax.legend(loc='upper right')
    plt.savefig('./result/loss_dagmm_{}.png'.format(n_gmm))