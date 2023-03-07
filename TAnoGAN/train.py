import torch
import torch.nn as nn
import logging
import wandb
import time
import pdb
import torch.nn.init as init
from torch.autograd import Variable
from collections import OrderedDict
import os
import json

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

def train(gen, dis, train_loader, 
        optimizer_G, optimizer_D, criterion, device,
        log_interval
        ):
    
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    acc_d_real_m = AverageMeter()
    acc_d_fake_m = AverageMeter()
    acc_g_m = AverageMeter()

    losses_d_real_m = AverageMeter()
    losses_d_fake_m = AverageMeter()
    losses_g_m = AverageMeter()
    losses_d_m = AverageMeter()

    end = time.time()

    dis.train()
    gen.train()

    optimizer_G.zero_grad()
    optimizer_D.zero_grad()

    for idx, (xx, _) in enumerate(train_loader):

        # Train Discriminator
        
        batch_size, window_size, in_dim = xx.size(0), xx.size(1), xx.size(2)

        optimizer_D.zero_grad()

        label = torch.ones((batch_size, window_size, in_dim)).to(device)
        label_size = label.flatten().shape[0]

        xx = xx.to(device)

        h_0, c_0 = torch.zeros(1, batch_size, dis.hidden_size).to(device), torch.zeros(1, batch_size, dis.hidden_size).to(device)
        h_g_0, c_g_0 = torch.zeros(1, batch_size, 32).to(device), torch.zeros(1, batch_size, 32).to(device)

        output,_ = dis(xx, h_0, c_0)
        
        loss_D_real = criterion(output, label)
        loss_D_real.backward()

        losses_d_real_m.update(loss_D_real.item())

        optimizer_D.step()

        preds = torch.round(output).detach().cpu()

        acc_d_real_m.update(label.detach().cpu().eq(preds).sum().item()/label_size, n=label_size)

        noise = Variable(init.normal(torch.Tensor(batch_size,window_size,1),mean=0,std=0.1)).to(device)
        fake, _ = gen(noise, h_g_0, c_g_0)

        output, _ = dis(fake.detach(), h_0, c_0)

        label = torch.zeros((batch_size, window_size, in_dim)).to(device)
        
        loss_d_fake = criterion(output, label)
        loss_d_fake.backward()

        losses_d_fake_m.update(loss_d_fake.item())
        loss_dis = loss_d_fake + loss_D_real
        optimizer_D.step()

        losses_d_m.update(loss_dis.item())

        preds = torch.round(output).detach().cpu()

        acc_d_fake_m.update(label.detach().cpu().eq(preds).sum().item()/label_size, n=label_size)

        # Train Generator
        optimizer_G.zero_grad()
        noise = Variable(init.normal(torch.Tensor(batch_size,window_size,1),mean=0,std=0.1)).to(device)
        fake, _ = gen(noise, h_g_0, c_g_0)
        
        label = torch.ones((batch_size, window_size, in_dim)).to(device)
        output, _ = dis(fake, h_0, c_0)

        loss_g = criterion(output, label)
        loss_g.backward()

        losses_g_m.update(loss_g.item())

        optimizer_G.step()

        preds = torch.round(output).detach().cpu()
        acc_g_m.update(label.detach().cpu().eq(preds).sum().item()/label_size, n=label_size)

        if idx % log_interval == 0 and idx != 0:

            _logger.info('TRAIN Iteration: [{:>4d}/{}] \n'
                        'Loss D Real: {losses_d_real.val:>6.4f} ({losses_d_real.avg:>6.4f}) '
                        'Acc D Real: {acc_d_real.avg:.3%} \n'
                        'Loss D Fake: {losses_d_fake.val:>6.4f} ({losses_d_fake.avg:>6.4f}) '
                        'Acc D Fake: {acc_d_fake.avg:.3%} \n'
                        'Loss D: {loss_dis:.3f} \n'
                        'Loss G: {losses_g.val:>6.4f} ({losses_g.avg:>6.4f}) '
                        'Acc G: {acc_g.avg:.3%} \n'
                        'LR: {lr:.3e} \n'.format(
                        idx+1, len(train_loader), 
                        losses_d_real = losses_d_real_m,
                        acc_d_real = acc_d_real_m,
                        losses_d_fake = losses_d_fake_m,
                        acc_d_fake = acc_d_fake_m,
                        loss_dis = loss_dis.item(),
                        losses_g = losses_g_m,
                        acc_g = acc_g_m,
                        lr    = optimizer_D.param_groups[0]['lr'],
                        )
            )

        end = time.time()

    return OrderedDict([('acc_d_real',acc_d_real_m.avg), ('loss_d_real',losses_d_real_m.avg),
                        ('acc_d_fake',acc_d_fake_m.avg), ('loss_d_real',losses_d_fake_m.avg),
                        ('acc_g',acc_g_m.avg), ('loss_g',losses_g_m.avg), ('loss_d', losses_d_m.avg)
                        ])

def eval(gen, dis, val_loader, 
        criterion, device,
        log_interval
        ):
    
    correct = 0
    total = 0
    total_loss_dis = 0
    total_loss_gen = 0

    gen.eval()
    dis.eval()

    with torch.no_grad():
        for idx, (xx, _) in enumerate(val_loader):
        
            batch_size, window_size, in_dim = xx.size(0), xx.size(1), xx.size(2)

            h_0, c_0 = torch.zeros(1, batch_size, dis.hidden_size).to(device), torch.zeros(1, batch_size, dis.hidden_size).to(device)
            h_g_0, c_g_0 = torch.zeros(1, batch_size, 32).to(device), torch.zeros(1, batch_size, 32).to(device)

            label = torch.ones((batch_size, window_size, in_dim)).to(device)
            label_size = label.flatten().shape[0]

            xx = xx.to(device)
            output, _ = dis(xx, h_0, c_0)

            loss_D_real = criterion(output, label)
            
            noise = Variable(init.normal(torch.Tensor(batch_size, window_size, in_dim),mean=0,std=0.1)).to(device)
            fake, _ = gen(noise, h_g_0, c_g_0)

            output, _ = dis(fake.detach(), h_0, c_0)

            label = torch.zeros((batch_size, window_size, in_dim)).to(device)
            
            loss_d_fake = criterion(output, label)

            loss_dis = loss_d_fake + loss_D_real

            total_loss_dis += loss_dis.item()

            noise = Variable(init.normal(torch.Tensor(batch_size,window_size, in_dim),mean=0,std=0.1)).to(device)
            fake, _ = gen(noise, h_g_0, c_g_0)
            
            label = torch.ones((batch_size, window_size, in_dim)).to(device)
            output, _ = dis(fake, h_0, c_0)

            loss_g = criterion(output, label)

            total_loss_gen += loss_g.item()
        
            if idx % log_interval == 0 and idx != 0:
                _logger.info('TEST [{}/{}]: Discriminator Loss: {:.3f} | Generator Loss: {:.3f} | Avg: {:.3f} '.format(idx+1,
                                                                                                len(val_loader), 
                                                                                                total_loss_dis/(idx+1),
                                                                                                total_loss_gen/(idx+1),
                                                                                                (total_loss_dis+total_loss_gen)/(idx+1)))

    return OrderedDict([('Dis_Loss',total_loss_dis/len(val_loader)), ('Gen_Loss',total_loss_gen/len(val_loader))])

def fit(gen, dis, train_loader, val_loader, criterion, optimizer_g, optimizer_d, 
        epochs, save_dir, log_interval, eval_interval, device, use_wandb,
        split):

    best_loss = 100000000000.0
    
    for epoch in range(epochs):
        _logger.info(f'\nEpoch: {epoch+1}/{epochs}')

        train_metrics = train(
                            gen = gen, dis = dis, train_loader = train_loader, optimizer_G = optimizer_g,
                            optimizer_D = optimizer_d, criterion = criterion, device = device, log_interval = log_interval
                            )
        metrics = OrderedDict(lr=optimizer_d.param_groups[0]['lr'])
        metrics.update([('train_' + k, v) for k, v in train_metrics.items()])
        
        dis_loss_cand = train_metrics['loss_d']
        gen_loss_cand = train_metrics['loss_g']

        if split:
            eval_metrics = eval(
                                gen = gen, dis = dis, val_loader = val_loader, criterion = criterion,
                                device = device, log_interval = eval_interval
                                )

            metrics.update([('eval_' + k, v) for k, v in eval_metrics.items()])
            dis_loss_cand = eval_metrics['Dis_Loss']
            gen_loss_cand = eval_metrics['Gen_Loss']

        if use_wandb:
            wandb.log(metrics, step=epoch+1)
    
        if best_loss > (dis_loss_cand+gen_loss_cand)/2:
            
            state = {'best_epoch':epoch+1, 
                    'best_Dis_loss':dis_loss_cand,
                    'best_Gen_loss':gen_loss_cand,
                    }

            json.dump(state, open(os.path.join(save_dir, f'best_results.json'),'w'), indent=4)
            
            torch.save(gen.state_dict(), os.path.join(save_dir, f'gen_best_model.pt'))
            torch.save(dis.state_dict(), os.path.join(save_dir, f'dis_best_model.pt'))
            
            _logger.info('Best Loss {:.3f} to {:.3f}'.format(best_loss, (dis_loss_cand+gen_loss_cand)/2))

            best_loss = (dis_loss_cand+gen_loss_cand)/2

    torch.save(gen.state_dict(), os.path.join(save_dir, f'gen_last_model.pt'))
    torch.save(dis.state_dict(), os.path.join(save_dir, f'dis_last_model.pt'))
            

    _logger.info('Best Metric: At {} Epoch Gen {:.3f} Dis'.format(state['best_epoch'],
                                                                                 state['best_Gen_loss'], 
                                                                                state['best_Dis_loss'],
                                                                                ))
