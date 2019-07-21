from torch import nn
import torch.nn.functional as F
import torch
from math import ceil
from pdb import set_trace

class vae(nn.Module):
    def __init__(self, encoder = None, decoder = None):
        super().__init__()
        self.encoder = tfattentionencoder() if encoder == None else encoder
        self.decoder = tfattentiondecoder() if decoder == None else decoder
    
    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        esp = torch.randn(*mu.size(), device = std.device)
        return mu + std * esp

    def bottleneck(self, h):
        mu, logvar = h[:, 0], h[:, 1]
        feature = self.reparameterize(mu, logvar)
        return feature, mu, logvar

    def forward(self, x):
        _, b, w, h = x.size()
        z1, z2 = self.encoder(x.view(-1, 1, w, h))
        feature, mu, logvar = self.bottleneck(z2)
        x_pred = self.decoder(feature.unsqueeze(-1).unsqueeze(-1))
        return x_pred.view(-1, b, w, h), z1, mu, logvar

class encoder(nn.Module):
    def __init__(self, filter_num, latent_size):
        super().__init__()
        self.filter_num = filter_num
        self.latent_size = latent_size
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, self.filter_num, (3,4), (1,2)),# self.filter_num, 19, 19
            nn.ReLU(inplace=True),
            )
        self.conv2 = nn.Sequential(
            nn.Conv2d(self.filter_num, self.filter_num * 2, 3, 2),# self.filter_num*2, 9, 9
            nn.ReLU(inplace=True),
        )
        self.conv3 =  nn.Sequential( 
            nn.Conv2d(self.filter_num * 2, self.filter_num * 4, 3, 2),# self.filter_num*4, 4, 4
            nn.ReLU(inplace=True)
            )#self.latent_size * 2, 1, 1
        self.conv4 = nn.Conv2d(self.filter_num * 4, self.latent_size * 2, 4, 1) 
    def forward(self,x):
        z1 = self.conv1(x)
        z1 = self.conv2(z1)
        z2 = self.conv3(z1)#4?
        z2 = self.conv4(z2)
        z2 = z2.view(-1, 2, self.latent_size)
        return z1, z2

class decoder(nn.Module):
    def __init__(self, filter_num, latent_size):
        super().__init__()
        self.filter_num = filter_num
        self.latent_size = latent_size
        self.convt1 = nn.Sequential(
            nn.ConvTranspose2d(latent_size, self.filter_num * 4, 4, 1),# self.filter_num*4, 4, 4
            nn.ReLU(inplace=True),
            )
        self.convt2 = nn.Sequential(    
            nn.ConvTranspose2d(self.filter_num * 4, self.filter_num * 2, 3, 2), # self.filter_num*4, 9, 9
            nn.ReLU(inplace=True),
            )
        self.convt3 = nn.Sequential(
            nn.ConvTranspose2d(self.filter_num * 2, self.filter_num, 3, 2), # self.filter_num*4, 19, 19
            nn.ReLU(inplace=True),
        )
        self.convt4 = nn.Sequential(
            nn.ConvTranspose2d(self.filter_num, 1, (3,4), (1,2)), # self.filter_num*4, 21, 40
            nn.Sigmoid()
        )
    def forward(self,x):
        x = self.convt2(self.convt1(x))
        x = self.convt4(  self.convt3( x ) )
        return x 
