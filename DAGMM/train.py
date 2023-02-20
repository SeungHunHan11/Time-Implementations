from dataset import TSDataset, TSLoader
import logging
import wandb
import pandas as pd
import time
import os
import json
import torch
from collections import OrderedDict
from torch.nn.utils import clip_grad_norm_
from utils import loss_func


_logger = logging.getLogger('train')

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def train(trainloader, model, device, criterion, optimizer, log_interval: int, lambda1, lambda2):
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    losses_m = AverageMeter()
    
    end = time.time()

    model.train()

    for idx, (xx,_) in enumerate(trainloader):
        
        optimizer.zero_grad()

        data_time_m.update(time.time() - end)

        xx = xx.to(device)
        z_c, reconstructed, z, gamma = model(xx)
        
        loss, recon_error, e, p = criterion(z_c.cpu(), reconstructed.cpu(), gamma.cpu(), z.cpu(), lambda1, lambda2)

        loss.backward()
        losses_m.update(loss.item())

        clip_grad_norm_(model.parameters(), 5)
        optimizer.step()

        batch_time_m.update(time.time() - end)

        if idx % log_interval == 0 and idx != 0: 

            _logger.info('TRAIN [{:>4d}/{}] Loss: {loss.val:>6.4f} ({loss.avg:>6.4f}) '
            'LR: {lr:.3e} '
            'Time: {batch_time.val:.3f}s, {rate:>7.2f}/s ({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s) '
            'Data: {data_time.val:.3f} ({data_time.avg:.3f})'.format(
            idx+1, len(trainloader), 
            loss       = losses_m, 
            lr         = optimizer.param_groups[0]['lr'],
            batch_time = batch_time_m,
            rate       = xx.size(0) / batch_time_m.val,
            rate_avg   = xx.size(0) / batch_time_m.avg,
            data_time  = data_time_m))
        
        end = time.time()

    return OrderedDict([('loss',losses_m.avg)])

def test(model, loader, criterion, device, log_interval: int, lambda1, lambda2):
    
    model.eval()
    total_loss = 0
    recon_loss = 0

    with torch.no_grad():

        for idx, (xx,_) in enumerate(loader):
            
            xx = xx.to(device)
            z_c, reconstructed, z, gamma = model(xx)
            
            loss, recon_error, e, p = criterion(z_c.cpu(), reconstructed.cpu(), gamma.cpu(), z.cpu(), lambda1, lambda2)

            total_loss += loss.item()
            recon_loss += recon_error.item()

            if idx % log_interval == 0 and idx != 0: 
                _logger.info('TEST [%d/%d]: Loss: %.3f' % 
                            (idx+1, len(loader), total_loss/(idx+1)))

    return OrderedDict([('loss',total_loss/len(loader))])

def fit(model, train_loader, val_loader, criterion, optimizer, scheduler, device,
    save_dir: str, use_wandb, lambda1: float = 0.001, lambda2: float = 0.005, 
    epochs: int = 50, log_interval: int = 1, eval_interval: int = 1000):

    os.makedirs(save_dir, exist_ok=True)

    best_loss = 1e10

    for epoch in range(epochs):
        _logger.info(f'\nEpoch: {epoch+1}/{epochs}')
        
        train_metrics = train(train_loader, model, device, criterion, optimizer, log_interval, lambda1, lambda2)
        eval_metrics = test(model, val_loader, criterion, device, eval_interval, lambda1, lambda2)

        metrics = OrderedDict([('train_' + k, v) for k, v in train_metrics.items()])
        metrics.update([('eval_' + k, v) for k, v in eval_metrics.items()])

        if use_wandb:
            wandb.log(metrics, step=epoch+1)

        if scheduler:
            
            scheduler.step()
        
        if eval_metrics['loss']<best_loss:

            state = {'best_epoch':epoch+1, 'best_loss':eval_metrics['loss']}

            json.dump(state, open(os.path.join(save_dir, f'best_results.json'),'w'), indent=4)

            torch.save(model.state_dict(), os.path.join(save_dir,'best_model.pt'))
            
            _logger.info('Best loss {0:.3} to {1:.3}'.format(best_loss, eval_metrics['loss']))

            best_loss=eval_metrics['loss']

    _logger.info('Best Metric: {0:.3} (epoch {1:})'.format(state['best_loss'], state['best_epoch']))