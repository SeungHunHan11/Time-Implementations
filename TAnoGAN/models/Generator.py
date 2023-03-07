import torch.nn as nn
import torch

class Gen(nn.Module):

    def __init__(self, in_dim, window_size):

        super(Gen,self).__init__()

        self.in_dim = in_dim
        self.window_size = window_size
        self.layer1 = nn.LSTM(in_dim, hidden_size = 32, num_layers = 1, batch_first = True) #stack of 3 LSTM with different hidden size
        self.layer2 = nn.LSTM(32, hidden_size = 64, num_layers = 1, batch_first = True)
        self.layer3 = nn.LSTM(64, hidden_size = 128, num_layers = 1, batch_first = True)
        self.FC = nn.Sequential(nn.Linear(128, in_dim), nn.Tanh())

    def forward(self, x, h_0, c_0):
        
        batch_size, window_size = x.size(0), x.size(1)
        out, _ = self.layer1(x, (h_0, c_0))
        out, _ = self.layer2(out)
        out, _ = self.layer3(out)

        FC_input = out.reshape(batch_size * window_size, 128)

        output = self.FC(FC_input)

        output = output.reshape(batch_size, window_size, self.in_dim)

        return output, out