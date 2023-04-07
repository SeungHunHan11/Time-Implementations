import torch
import torch.nn as nn

import sys
sys.path.append('/directory/')
from model.attention import Attention
from model.input_embedding import TSEmbedding
from model.series_attention import Series_Attention
from model.prior_attention import Prior_Attention

class EncoderBlock(nn.Module):
    def __init__(self, seq_len, attention, d_model, d_ff, dropout, activation, norm_type: str):
        super(EncoderBlock, self).__init__()

        d_ff = d_ff or 4 * d_model

        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1) # FFNN using Conv1d

        norm_size = d_model if norm_type == 'LayerNorm' else seq_len

        self.norm1 = __import__('torch.nn', fromlist = 'nn').__dict__[f'{norm_type}'](norm_size)
        self.norm2 = __import__('torch.nn', fromlist = 'nn').__dict__[f'{norm_type}'](norm_size)

        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU() if activation == 'relu' else nn.GELU()

    def forward(self, x, attn_mask = None):

        recon, series, prior, sigma = self.attention(x,x,x,  attn_mask = attn_mask)
        x = x + self.dropout(recon) # x += self.dropout(recon) makes an error
        x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(x.transpose(-1,1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1)) #back to original shape i.e. seq len, d_model

        return self.norm2(x + y), series, prior, sigma
    
class Encoder(nn.Module):

    def __init__(self, seq_len, attention_layers, normal_layer = None):

        super(Encoder, self).__init__()

        self.attn_layers = nn.ModuleList(attention_layers)
        self.normal_layer = normal_layer

    def forward(self, x, attn_mask = None):
        series_list = []
        prior_list = []
        sigma_list = []
        
        for attn_layers in self.attn_layers:
            x, series, prior, sigma = attn_layers(x, attn_mask=attn_mask)
            series_list.append(series)
            prior_list.append(prior)
            sigma_list.append(sigma)

        if self.normal_layer is not None:  
            x = self.normal_layer(x)

        return x, series_list, prior_list, sigma_list

class Anomaly_Transformer(nn.Module):
    def __init__(self, window_size, c_in, c_out, d_model, n_head, num_layers, d_ff,
                 dropout, activation, norm_type, pos_type, emb_type, device, output_attention = True,
                 drop_pos = False):
                 
        super(Anomaly_Transformer, self).__init__()
        self.output_attention = output_attention

        prior_attention = Prior_Attention(window_size = window_size, device = device)
        series_attention = Series_Attention(window_size = window_size, mask_flag = False, scale = None, attention_dropout = 0.3)
        
        attention = Attention(prior_attention = prior_attention,
                              series_attention = series_attention, 
                              d_model = d_model, 
                              n_heads = n_head, 
                              d_key = None, 
                              d_value = None)

        self.TSEmbedding = TSEmbedding(c_in = c_in, 
                                       d_model = d_model, 
                                       pos_type = pos_type, 
                                       emb_type = emb_type, 
                                       dropout = dropout,
                                       drop_pos = drop_pos)

        norm_size = d_model if norm_type == 'LayerNorm' else window_size

        norm_layer = __import__('torch.nn', fromlist = 'nn').__dict__[f'{norm_type}'](norm_size)
        
        self.encoder = Encoder(seq_len = window_size, 
                               attention_layers = [
                                EncoderBlock(
                                seq_len = window_size, 
                                attention = attention,
                                d_model = d_model, 
                                d_ff = d_ff, 
                                dropout = dropout, 
                                activation = activation, 
                                norm_type = norm_type, 
                               ) for l in range(num_layers)
                               ],
                               normal_layer=norm_layer
                               )

        self.projection = nn.Linear(d_model, c_out, bias = True)
        self.relu = nn.ReLU()

    def forward(self, x):
        enc_out = self.TSEmbedding(x)
        recon_latent, series_list, prior_list, sigma_list = self.encoder(enc_out, attn_mask = None)
        recon_x = self.relu(self.projection(recon_latent))

        if self.output_attention:
            return recon_x, series_list, prior_list, sigma_list 
        else:
            return recon_x

# sample = torch.zeros((1,100,38))


# prior_attention = Prior_Attention(window_size = 100, device = torch.device('cpu'))
# series_attention = Series_Attention(window_size = 100, mask_flag = True, scale = None, attention_dropout = 0.3)
# attention = Attention(prior_attention = prior_attention, series_attention = series_attention, d_model = 126, n_head = 6, d_key = None, d_value = None)
# model = Anomaly_Transformer(window_size = 100, c_in = 38, c_out = 38, d_model = 126, n_head = 6, num_layers = 3, d_ff = 100,
#            dropout = 0.1, activation = 'relu', norm_type = 'BatchNorm1d', pos_type = 'learnable', emb_type = 'Conv1d', 
#            device = torch.device('cpu'),
#             output_attention = True)
