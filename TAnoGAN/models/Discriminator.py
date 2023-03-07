import torch.nn as nn
import torch

class Dis(nn.Module):
    def __init__(self, in_dim, hidden_size):
        super(Dis, self).__init__()

        self.in_dim = in_dim
        self.hidden_size = hidden_size

        self.layer1 = nn.LSTM(in_dim, hidden_size=hidden_size, num_layers = 1, batch_first = True)
        self.linear = nn.Sequential(nn.Linear(hidden_size, 1), nn.Sigmoid())

    def forward(self, x, h_0, c_0):

        batch_size, window_size = x.size(0), x.size(1)
        out, _ = self.layer1(x, (h_0, c_0))
        output = out.reshape(batch_size * window_size, self.hidden_size)

        output = self.linear(output).reshape(batch_size, window_size, 1)

        return output, out