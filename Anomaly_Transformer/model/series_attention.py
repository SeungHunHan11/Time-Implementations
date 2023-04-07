import torch
import torch.nn as nn 
import math
import numpy as np

class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask

class Series_Attention(nn.Module):

    def __init__(self, window_size, mask_flag=False, scale=None, attention_dropout=0.0):
        super(Series_Attention, self).__init__()

        self.scale = scale
        self.mask_flag = mask_flag
        self.dropout = nn.Dropout(attention_dropout)


    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape # Batch size, Seq len, num head, d_model/num head
        S = values.shape[1] # Values: B, S, H, E
        scale = self.scale or 1. / math.sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys) #Get Attention Score. L = S for self-attention

        if self.mask_flag: # Use causal masking or not
            if attn_mask is None: 
                attn_mask = TriangularCausalMask(B, L, device=queries.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)
        
        attn = scale * scores

        series = self.dropout(torch.softmax(attn, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", series, values) #Multiply attention score and value matrix

        # assert self.window_size == sigma.shape[1]

        # sigma = sigma.transpose(1,2) # B L H ->  B H L
        # sigma = torch.sigmoid(sigma * 5) + 1e-5
        # sigma = torch.pow(3, sigma) - 1
        # window_size = attn.shape[-1]

        # sigma = sigma.unsqueeze(-1).repeat(1, 1, 1, window_size)  # B H L L
        # prior = self.distances.unsqueeze(0).unsqueeze(0).repeat(sigma.shape[0], sigma.shape[1], 1, 1)
        # prior = 1.0 / (math.sqrt(2 * math.pi) * sigma) * torch.exp(-prior ** 2 / 2 / (sigma ** 2))
        
        return (V.contiguous(), series)