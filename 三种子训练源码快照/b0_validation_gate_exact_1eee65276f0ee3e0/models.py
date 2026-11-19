"""Shared CI temporal residual-MLP skeleton, without SOFTS STAR mechanisms."""
import torch
from torch import nn

class ResidualFFN(nn.Module):
    def __init__(self, dim, hidden, dropout):
        super().__init__()
        self.ffn=nn.Sequential(nn.Linear(dim,hidden),nn.GELU(),nn.Dropout(dropout),
                               nn.Linear(hidden,dim),nn.Dropout(dropout))
        self.norm=nn.LayerNorm(dim)
    def forward(self,z):
        return self.norm(z+self.ffn(z))

class Baseline(nn.Module):
    def __init__(self, config, channels):
        super().__init__()
        self.length=config.get('length',96); self.horizon=config.get('horizon',96)
        dim=config.get('dim',128); hidden=config.get('ffn_dim',dim*4)
        self.embedding=nn.Sequential(nn.Linear(self.length,dim),nn.Dropout(config.get('dropout',0.0)))
        self.encoder=nn.Sequential(*[ResidualFFN(dim,hidden,config.get('dropout',0.0))
                                    for _ in range(config.get('layers',2))])
        self.head=nn.Linear(dim,self.horizon)

    def normalize(self,x):
        mu=x.mean(1,keepdim=True).detach()
        sigma=(x.var(1,keepdim=True,unbiased=False)+1e-5).sqrt().detach()
        return (x-mu)/sigma,mu,sigma

    def forward(self,x):
        xn,mu,sigma=self.normalize(x)
        z=self.encoder(self.embedding(xn.transpose(1,2)))
        return self.head(z).transpose(1,2)*sigma+mu

def create_model(config, channels):
    if config['model']=='B0': return Baseline(config,channels)
    if config['model']=='I1_context':
        from i1_context import Model
        return Model(config,channels)
    raise ValueError('Unknown model '+config['model'])
