import torch
import argparse

def kl_loss(p, q):
    res = p * (torch.log(p + 0.0001) - torch.log(q + 0.0001))
    return torch.mean(torch.sum(res, dim=-1), dim=1)


def min_max_loss(prior, series, window_size,temperature: int = 1):

    maximum_phase_loss = 0.0
    minimum_phase_loss = 0.0

    for n in range(len(prior)):

        series_dist = series[n]
        prior_dist = (prior[n] / torch.unsqueeze(torch.sum(prior[n], dim=-1), dim=-1).repeat(1, 1, 1,window_size))
        
        maximum_phase_loss = maximum_phase_loss +  (torch.mean(kl_loss(series_dist, prior_dist.detach()))+torch.mean(kl_loss(prior_dist.detach(),series_dist)))*temperature
        minimum_phase_loss = minimum_phase_loss + (torch.mean(kl_loss(series_dist.detach(), prior_dist))+torch.mean(kl_loss(prior_dist,series_dist.detach())))*temperature

    maximum_phase_loss = maximum_phase_loss / len(prior)
    minimum_phase_loss = minimum_phase_loss / len(prior)

    return maximum_phase_loss, minimum_phase_loss

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')