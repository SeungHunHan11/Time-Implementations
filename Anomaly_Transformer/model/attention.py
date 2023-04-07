import torch
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self, prior_attention, series_attention, 
                 d_model, n_heads,
                 d_key=None,
                 d_value=None
                 ):
        super(Attention, self).__init__()

        d_key, d_values = d_key or d_model // n_heads, d_value or d_model // n_heads
        
        self.prior_attention = prior_attention
        self.series_attention = series_attention

        self.query_projection = nn.Linear(d_model,
                                          d_key * n_heads)
        self.key_projection = nn.Linear(d_model,
                                        d_key * n_heads)
        self.value_projection = nn.Linear(d_model,
                                          d_values * n_heads)
        self.sigma_projection = nn.Linear(d_model,
                                          n_heads)
        
        self.out_projection = nn.Linear(d_values * n_heads, d_model)

        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask = None):
        B, L, _ = queries.shape # Batch size, seq len, d_model
        _, S, _ = keys.shape # Batch size, seq len, d_model
        H = self.n_heads
        x = queries.clone()

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)
        sigma = self.sigma_projection(x).view(B, L, H)

        out, series = self.series_attention(
            queries,
            keys,
            values,
            attn_mask
        )

        prior, sigma = self.prior_attention(sigma)

        out = out.view(B, L, -1) #Batch size, Seq len, d_model

        return self.out_projection(out), series, prior, sigma






# (torch.mean(my_kl_loss(series[u], (
#                         prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
#                                                                                                self.win_size)).detach())) + torch.mean(
#                     my_kl_loss(
#                         (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
#                                                                                                 self.win_size)).detach(),
#                         series[u])) )