import torch
from utils import kl_loss, min_max_loss
import time
import wandb
import os
import numpy as np
import torch.nn as nn
from inference import get_energy, evaluation
import json
from sklearn.metrics import precision_recall_fscore_support, f1_score
from sklearn.metrics import accuracy_score

def train(train_loader, model, optimizer, criterion, device, Lambda):

    model.train()

    loss_min_list = 0.0
    loss_max_list = 0.0

    for idx, (xx, _) in enumerate(train_loader):
        with torch.autograd.set_detect_anomaly(True):

            xx = xx.to(device)
            
            window_size = xx.shape[1]

            output, series, prior, _ = model(xx)

            maximum_phase_loss, minimum_phase_loss = min_max_loss(prior = prior, series = series, window_size = window_size, temperature = 1)

            reconstruction_loss = criterion(output, xx)
            loss_min = reconstruction_loss + Lambda * minimum_phase_loss
            loss_max = reconstruction_loss - Lambda * maximum_phase_loss
        
            loss_min_list = (loss_min_list + loss_min).item()
            loss_max_list = (loss_max_list + loss_max).item()

            if (idx + 1) % 1 == 0:
                print('At [{}/{}] \n Minimum train loss : {:0.4f} Average: ({:0.4f})   Maximum train loss : {:0.4f}, Average: ({:0.4f})'.format(
                    idx+1,len(train_loader),loss_min,loss_min_list/(idx+1),loss_max, loss_max_list/(idx+1)
                ))
            optimizer.zero_grad()

            loss_max.backward(retain_graph = True)

            loss_min.backward()

            optimizer.step()

    return loss_min_list/(idx+1), loss_max_list/(idx+1)
    
def eval(data_loader, model, criterion, device, Lambda):

    model.eval()
    loss_min_list = 0.0
    loss_max_list = 0.0

    with torch.no_grad():
            
        for idx, (xx, _) in enumerate(data_loader):
            xx = xx.to(device)
            window_size = xx.shape[1]

            output, series, prior, _ = model(xx)
            
            maximum_phase_loss, minimum_phase_loss = min_max_loss(prior = prior, series = series, window_size = window_size, temperature = 1)

            reconstruction_loss = criterion(output, xx)
            loss_max = reconstruction_loss - Lambda * maximum_phase_loss
            loss_min = reconstruction_loss + Lambda * minimum_phase_loss
            
            loss_min_list += loss_min
            loss_max_list += loss_max

            if (idx + 1) % 1 == 0:
                print('At [{}/{}] \n Minimum Validation loss : {:0.4f} Average: ({:0.4f})   Maximum Validation loss : {:0.4f}, Average: ({:0.4f})'.format(
                    idx+1,len(data_loader),loss_min,loss_min_list/(idx+1),loss_max, loss_max_list/(idx+1)
                ))

    return loss_min_list/(idx+1), loss_max_list/(idx+1)

def fit(train_loader, val_loader, model, 
        optimizer, criterion, scheduler, 
        device, Lambda, use_wandb,
        save_dir, epochs, temperature = 50, anomaly_ratio = 4.00, 
        threshold_loader = None, test_loader = None
        ):
    
    best_loss_min = 0
    best_loss_max = 0

    for epoch in range(epochs):
        train_loss_min, train_loss_max = train(train_loader, model, optimizer, criterion, device, Lambda)
        val_loss_min, val_loss_max = eval(val_loader, model, criterion, device, Lambda)

        metrics = {'Train_loss_min': train_loss_min, 'Train_loss_max':train_loss_max,
                   'Val_loss_min':val_loss_min, 'Val_loss_max': val_loss_max
                   }

        print(f'At [{epoch}/{epochs}] Maximum phase Loss is {val_loss_max}. Minimum phase Loss is {val_loss_min}')
        
        if use_wandb:
            wandb.log(metrics, step = epoch)

        if scheduler:
            scheduler.step()        

        if threshold_loader is not None:

            criterion_infer = nn.MSELoss(reduction='none')

            model.eval()

            with torch.no_grad():
                train_energy = get_energy(model = model, 
                                        criterion = criterion_infer, 
                                        data_loader = train_loader, 
                                        device = device,
                                        temperature = temperature, 
                                        anomaly_ratio = anomaly_ratio,
                                        train_energy = None
                                        )
                
                threshold = get_energy(model = model, 
                                        criterion = criterion_infer, 
                                        data_loader = threshold_loader, 
                                        device = device,
                                        temperature = temperature, 
                                        anomaly_ratio = anomaly_ratio,
                                        train_energy = train_energy
                                        )

                pred, ground_truth= evaluation(
                                            loader = threshold_loader,
                                            device = device,
                                            criterion = criterion_infer,
                                            model = model, 
                                            temperature = temperature,
                                            threshold = threshold
                                            )

            accuracy = accuracy_score(ground_truth, pred)
            precision, recall, f_score, support = precision_recall_fscore_support(ground_truth, pred,
                                                                                average='binary')
            f1_sc = f1_score(ground_truth, pred)

            eval_result = {
                        'Epoch' : epoch,
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
            
            json.dump(eval_result, open(os.path.join(save_dir, f"epoch_{epoch}_test_result.json"),'w'), indent='\t')

        if val_loss_min > best_loss_min and -1*val_loss_max > best_loss_max:
            
            if use_wandb:
                wandb.log(eval_result, step = epoch)

            print(f'At epoch {epoch} New Best score updated from Min_phase: {best_loss_min} Max Phase: {best_loss_max} to Min_phase: {-1*val_loss_min}, Max Phase: {-1*val_loss_max}')

            best_loss_min = val_loss_min
            best_loss_max = -1*val_loss_max

            torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pt'))

        torch.save(model.state_dict(), os.path.join(save_dir, 'latest_model.pt'))
