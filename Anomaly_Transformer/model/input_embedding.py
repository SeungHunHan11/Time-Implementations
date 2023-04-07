import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]

class PositionalEmbedding(nn.Module):
    # https://github.com/CyberZHG/torch-position-embedding/blob/master/torch_position_embedding/position_embedding.py

    def __init__(self,
                d_model: int,
                num_embeddings: int = 5000
                ):
        
        super(PositionalEmbedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.d_model = d_model

        self.weight = nn.Parameter(torch.Tensor(num_embeddings, d_model), requires_grad = True)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_normal_(self.weight)

    def forward(self, x):
        batch_size, seq_len = x.size()[:2]
        embeddings = self.weight[:seq_len, :].view(1, seq_len, self.d_model)

        return x + embeddings
    
class ConvTokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(ConvTokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x

class LinearTokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(LinearTokenEmbedding, self).__init__()
        self.linear = nn.Linear(c_in, d_model)

    def forward(self, x):
        return self.linear(x)
    
class TSEmbedding(nn.Module):
    def __init__(self, c_in, d_model, pos_type:str, emb_type: str, dropout: float = 0.4,
                 drop_pos: bool = False):
        super(TSEmbedding, self).__init__()

        if pos_type == 'learnable':
            self.pos = PositionalEmbedding(d_model = d_model)
        else:
            self.pos = PositionalEncoding(d_model = d_model)

        if emb_type == 'Conv1d':
            self.emb = ConvTokenEmbedding(c_in = c_in, d_model = d_model)
        else:
            self.emb = LinearTokenEmbedding(c_in = c_in, d_model = d_model)

        self.dropout = nn.Dropout(p=dropout)

        self.drop_pos = drop_pos

    def forward(self, x):

        out = self.emb(x)

        if not self.drop_pos:
            out = self.pos(out)

        out = self.dropout(out)

        return out
    
# emb = TSEmbedding(38, 126, 'learnable', 'Conv1d', 0.4)
# emb(sample).shape