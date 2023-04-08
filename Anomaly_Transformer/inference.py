import torch
from utils import kl_loss
import numpy as np

def get_energy(
            model, criterion, data_loader, device,
            temperature, anomaly_ratio, train_energy = None, 
            ):

    attens_energy = []

    for i, (xx, _) in enumerate(data_loader):

        window_size = xx.shape[1]
        xx = xx.to(device)
        output, series, prior, _ = model(xx)
        loss = torch.mean(criterion(xx, output), dim=-1)
        
        series_loss = 0.0
        prior_loss = 0.0
        for u in range(len(prior)):
            if u == 0:
                series_loss = kl_loss(series[u], (
                        prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                window_size)).detach()) * temperature
                prior_loss = kl_loss(
                    (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                            window_size)),
                    series[u].detach()) * temperature
            else:
                series_loss += kl_loss(series[u], (
                        prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                window_size)).detach()) * temperature
                prior_loss += kl_loss(
                    (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                            window_size)),
                    series[u].detach()) * temperature
                    
        metric = torch.softmax((-series_loss-prior_loss), dim = -1)
        cri = metric * loss
        cri = cri.detach().cpu().numpy()
        attens_energy.append(cri)

    attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
    dataset_energy = np.array(attens_energy)

    if train_energy is None:

        print('Calculating train energy')
        train_energy = dataset_energy
        return train_energy
    
    else:
        print('Calculating threshold...')

        test_energy = dataset_energy
        combined_energy = np.concatenate([train_energy, test_energy], axis=0)
        thresh = np.percentile(combined_energy, 100 - anomaly_ratio)
        print("Threshold :", thresh)

        return thresh
    
def evaluation(loader, device, criterion, model, temperature, threshold):

    test_labels = []
    attens_energy = []

    for i, (xx, yy) in enumerate(loader):
        xx = xx.to(device)
        window_size = xx.shape[1]

        output, series, prior, _ = model(xx)

        loss = torch.mean(criterion(xx, output))

        series_loss = 0.0
        prior_loss = 0.0

        for u in range(len(prior)):
            if u == 0:
                series_loss = kl_loss(series[u], (
                        prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                window_size)).detach()) * temperature
                prior_loss = kl_loss(
                    (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                            window_size)),
                    series[u].detach()) * temperature
            else:
                series_loss += kl_loss(series[u], (
                        prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                window_size)).detach()) * temperature
                prior_loss += kl_loss(
                    (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                            window_size)),
                    series[u].detach()) * temperature
                    
        metric = torch.softmax((-series_loss-prior_loss), dim = -1)

        cri = metric * loss
        cri = cri.detach().cpu().numpy()
        attens_energy.append(cri)
        test_labels.append(yy)

    attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
    test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
    test_energy = np.array(attens_energy)
    test_labels = np.array(test_labels) 

    pred = (test_energy > threshold).astype(int)
    gt = test_labels.astype(int)

    print("pred:   ", pred.shape)
    print("gt:     ", gt.shape)

    # detection adjustment
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1

    pred = np.array(pred)
    gt = np.array(gt)

    print("pred: ", pred.shape)
    print("gt:   ", gt.shape)

    return pred, gt
