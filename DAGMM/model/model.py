import torch.nn as nn
from torch.nn.functional import cosine_similarity, pairwise_distance
import torch

class DAGMM(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, latent_dim, dropout,n_gmm):
        
        super(DAGMM,self).__init__()
        
        self.input_dim= input_dim
        self.hidden_dim= hidden_dim


        layers = []
        layers += [nn.Linear(input_dim, hidden_dim)]
        layers += [nn.Tanh()]        
        layers += [nn.Linear(hidden_dim, int(hidden_dim/2))]
        layers += [nn.Tanh()]        
        layers += [nn.Linear(int(hidden_dim/2), int(hidden_dim/3))]
        layers += [nn.Tanh()]        
        layers += [nn.Linear(int(hidden_dim/3), latent_dim)]

        self.encoder= nn.Sequential(*layers)

        layers = []
        layers += [nn.Linear(latent_dim, int(hidden_dim/3))]
        layers += [nn.Tanh()]        
        layers += [nn.Linear(int(hidden_dim/3), int(hidden_dim/2))]
        layers += [nn.Tanh()]        
        layers += [nn.Linear(int(hidden_dim/2), hidden_dim)]
        layers += [nn.Tanh()]        
        layers += [nn.Linear(hidden_dim, input_dim)]
        self.decoder = nn.Sequential(*layers)

        layers = []
        layers += [nn.Linear(latent_dim+2,int(hidden_dim/3))] #Latent vec +recon. error + sim. matrix

        layers += [nn.Tanh()]        
        layers += [nn.Dropout(p=dropout)]        
        layers += [nn.Linear(int(hidden_dim/3),n_gmm)]
        layers += [nn.Softmax(dim=1)]

        self.FC = nn.Sequential(*layers)

    def forward(self, x):

        z_c = self.encoder(x)
        reconstructed = self.decoder(z_c)
        sim_mat = cosine_similarity(x, reconstructed)
        dist = pairwise_distance(x, reconstructed, p = 2)
    
        z = torch.cat([z_c, sim_mat.unsqueeze(-1), dist.unsqueeze(-1)],dim=1)
        gamma = self.FC(z)

        return z_c, reconstructed, z, gamma
         