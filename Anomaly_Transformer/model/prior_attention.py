import torch.nn as nn
import torch
import math

class Prior_Attention(nn.Module):
    def __init__(self, window_size, device):
        super(Prior_Attention, self).__init__()
        self.window_size = window_size
        self.distances = torch.zeros((window_size, window_size))
        self.device = device

        for i in range(window_size):
            for j in range(window_size):
                self.distances[i][j] = abs(i - j)

        self.distances = self.distances.to(device)

    def forward(self, sigma):
        sigma = sigma.transpose(1,2) # B L H ->  B H L

        window_size = sigma.shape[-1]
        assert self.window_size == window_size

        sigma = torch.sigmoid(sigma * 5) + 1e-5
        sigma = sigma.unsqueeze(-1).repeat(1, 1, 1, window_size)  # B H L L
        prior = self.distances.unsqueeze(0).unsqueeze(0).repeat(sigma.shape[0], sigma.shape[1], 1, 1)
        prior = 1.0 / (math.sqrt(2 * math.pi) * sigma) * torch.exp(-prior ** 2 / 2 / (sigma ** 2))

        return prior, sigma