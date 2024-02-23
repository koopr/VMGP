import torch
import torch.nn as nn


class VMGP(nn.Module):   

    def __init__(self, input_dim, out_dim=1, latent_dim=32, dropout=None, hidden_dims = [2048,512], reg_scale = 4):  
        super(VMGP, self).__init__() 
        if dropout is None:
            dropout = .3 if input_dim > 64 else 0
        dim1 = hidden_dims[0]
        dim2 = hidden_dims[1]
        self.out_dim = out_dim
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, dim1),
            nn.BatchNorm1d(dim1),
            nn.LeakyReLU(),
            nn.Dropout(dropout),

            nn.Linear(dim1, dim2),
            nn.BatchNorm1d(dim2),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
        )
        self.fc_mus = nn.Linear(dim2, latent_dim)
        self.fc_vars = nn.Linear(dim2, latent_dim)
        self.decoder = nn.Sequential(
            
            nn.Linear(latent_dim, dim2),
            nn.BatchNorm1d(dim2),
            nn.LeakyReLU(),
            nn.Dropout(dropout),

            nn.Linear(dim2, dim1),
            nn.BatchNorm1d(dim1),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(dim1, input_dim),
        )
        self.regs = []
        for i in range(self.out_dim):
            self.regs.append(nn.Sequential(
                nn.Linear(latent_dim, reg_scale*latent_dim),
                nn.BatchNorm1d(reg_scale*latent_dim),
                nn.LeakyReLU(),
                nn.Linear(reg_scale*latent_dim, 1)
            ))
        self.regs = nn.ModuleList(self.regs)    

    def reg(self, X):
        return [self.regs[i](X) for i in range(self.out_dim)]
    
    def forward(self, X):
        encode = self.encoder(X)
        mu = self.fc_mus(encode)
        logvar = self.fc_vars(encode)
        std = torch.exp(logvar / 2)
        if not self.training:
            zs = mu
            return torch.cat(self.reg(zs),dim=1)
        else:
            std = std + 1e-7
            q = torch.distributions.Normal(mu, std)
            zs = q.rsample()
        y_hat = torch.cat(self.reg(zs),dim=1)
        return y_hat, self.decoder(zs), mu, logvar