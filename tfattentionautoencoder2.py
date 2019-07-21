from torch import nn
import torch.nn.functional as F
import torch
from math import ceil
from pdb import set_trace

class tfattentionvae(nn.Module):
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
        z1, z2, excitate = self.encoder(x.view(-1, 1, w, h))
        feature, mu, logvar = self.bottleneck(z2)
        x_pred = self.decoder(feature.unsqueeze(-1).unsqueeze(-1), excitate)
        return x_pred.view(-1, b, w, h), z1, mu, logvar

class tfattentionencoder(nn.Module):
    def __init__(self, filter_num, latent_size, detach):
        super().__init__()
        self.filter_num = filter_num
        self.latent_size = latent_size
        self.detach = detach
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
class tfattentiondecoder(nn.Module):
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

class TFSEnet(nn.Module):
    def __init__(self,insize, sesize):
        super().__init__()
        self.linear1_s = nn.Sequential( nn.Linear(insize[0],sesize[0]), nn.BatchNorm2d(1), nn.Dropout(p=0.2), nn.ReLU() ) 
        self.linear2_s = nn.Sequential( nn.Linear(insize[1],sesize[1]), nn.BatchNorm2d(1), nn.Dropout(p=0.2), nn.ReLU() )  
        self.linear1_u = nn.Sequential( nn.Linear(sesize[0],insize[0]), nn.BatchNorm2d(1), nn.Dropout(p=0.2), nn.Sigmoid() )
        self.linear2_u = nn.Sequential( nn.Linear(sesize[1],insize[1]), nn.BatchNorm2d(1), nn.Dropout(p=0.2), nn.Sigmoid() )

    def forward(self, x):
        _, c, _, _ = x.size()
        x = F.avg_pool2d(x.permute(0,2,1,3),(c,1)).permute(0,2,1,3)        
        x = self.linear2_s(x)
        x = self.linear1_s(x.permute(0,1,3,2)).permute(0,1,3,2)
        x = self.linear2_u(x)
        x = self.linear1_u(x.permute(0,1,3,2)).permute(0,1,3,2)
        return x
class TFattention(nn.Module):
    def __init__(self, insize):
        super().__init__()
        self.linear1 = nn.Sequential(nn.Linear(insize[0], insize[0]), nn.BatchNorm2d(1), nn.Dropout(p=0.2), nn.Sigmoid() )
        self.linear2 = nn.Sequential(nn.Linear(insize[1], insize[1]), nn.BatchNorm2d(1), nn.Dropout(p=0.2), nn.Sigmoid() )
    def forward(self, x):
        _, c, _, _ = x.size()
        x = F.avg_pool2d(x.permute(0,2,1,3) ,(c,1)).permute(0,2,1,3)        
        x = self.linear1(x.permute(0,1,3,2)).permute(0,1,3,2)
        x = self.linear2(x)
        return x

#*****************************23******************************************
class TFAencoder23(tfattentionencoder):
    def __init__(self, filter_num, latent_size, detach):
        super().__init__(filter_num, latent_size, detach)
        self.tfsequeeze1 = TFattention(insize = [19, 19])
        self.tfsequeeze2 = TFattention(insize = [9, 9])

    def forward(self, x):
        z1 = self.conv1( x )
        excitate1 = self.tfsequeeze1(z1).repeat(1, self.filter_num, 1, 1)
        z1 = self.conv2( torch.mul(z1, excitate1) )
        excitate2 = self.tfsequeeze2(z1).repeat(1, self.filter_num * 2, 1, 1)
        z1 = torch.mul(z1, excitate2)
        z2 = self.conv3(z1)#4?
        z2 = self.conv4(z2)
        z2 = z2.view(-1, 2, self.latent_size)
        excitate = [excitate1.detach(), excitate2.detach()] if self.detach == True else [excitate1, excitate2]
        return z1, z2, excitate 
    
class TFAdecoder23(tfattentiondecoder):
    def __init__(self, filter_num, latent_size):
        super().__init__(filter_num , latent_size)

    def forward(self, x, excitate):
        x = self.convt2(self.convt1(x))
        x = self.convt4( torch.div( self.convt3( torch.div(x, excitate[1]) ), excitate[0] ) )
        return x 
#*****************************24******************************************
class TFAencoder24(tfattentionencoder):
    def __init__(self, filter_num, latent_size, detach):
        super().__init__(filter_num, latent_size, detach)
        self.tfsequeeze1 = TFattention(insize = [19, 19])
        self.tfsequeeze2 = TFattention(insize = [4, 4])

    def forward(self, x):
        z1 = self.conv1(x)#19
        excitate1 = self.tfsequeeze1(z1).repeat(1, self.filter_num, 1, 1)
        z1 = self.conv2( torch.mul(z1, excitate1) )#9
        z2 = self.conv3(z1)#4?
        excitate2 = self.tfsequeeze2(z2).repeat(1, self.filter_num * 4, 1, 1)
        z2 = torch.mul(z2, excitate2)
        z2 = self.conv4(z2)
        z2 = z2.view(-1, 2, self.latent_size)
        excitate = [excitate1.detach(), excitate2.detach()] if self.detach == True else [excitate1, excitate2]
        return z1, z2, excitate 
    
class TFAdecoder24(tfattentiondecoder):
    def __init__(self, filter_num, latent_size):
        super().__init__(filter_num , latent_size)

    def forward(self, x, excitate):
        x = self.convt2(torch.div(self.convt1(x),excitate[1]))
        x = self.convt4( torch.div( self.convt3(x), excitate[0] ) )
        return x 
#*****************************14******************************************
class TFAencoder14(tfattentionencoder):
    def __init__(self, filter_num, latent_size, detach):
        super().__init__(filter_num, latent_size, detach)
        self.tfsequeeze1 = TFattention(insize = [21, 40])
        self.tfsequeeze2 = TFattention(insize = [4, 4])

    def forward(self, x):
        excitate1 = self.tfsequeeze1(x)
        z1 = self.conv1( torch.mul(x, excitate1) )#19
        z1 = self.conv2(z1)#9
        z2 = self.conv3(z1)
        excitate2 = self.tfsequeeze2(z2).repeat(1, self.filter_num * 4, 1, 1)
        z2 = torch.mul(z2, excitate2)
        z2 = self.conv4(z2)
        z2 = z2.view(-1, 2, self.latent_size)
        excitate = [excitate1.detach(), excitate2.detach()] if self.detach == True else [excitate1, excitate2]
        return z1, z2, excitate 
    
class TFAdecoder14(tfattentiondecoder):
    def __init__(self, filter_num, latent_size):
        super().__init__(filter_num , latent_size)

    def forward(self, x, excitate):
        x = self.convt2(torch.div(self.convt1(x),excitate[1]))
        x = torch.div( self.convt4(self.convt3(x)), excitate[0] )
        return x 
#*****************************13******************************************
class TFAencoder13(tfattentionencoder):
    def __init__(self, filter_num, latent_size, detach):
        super().__init__(filter_num, latent_size, detach)
        self.tfsequeeze1 = TFattention(insize = [21, 40])
        self.tfsequeeze2 = TFattention(insize = [9, 9])

    def forward(self, x):
        excitate1 = self.tfsequeeze1(x)
        z1 = self.conv1(torch.mul(x, excitate1))
        z1 = self.conv2(z1)
        excitate2 = self.tfsequeeze2(z1).repeat(1, self.filter_num * 2, 1, 1)
        z1 = torch.mul(z1, excitate2)
        z2 = self.conv3( z1 )
        z2 = self.conv4(z2)
        z2 = z2.view(-1, 2, self.latent_size)
        excitate = [excitate1.detach(), excitate2.detach()] if self.detach == True else [excitate1, excitate2]
        return z1, z2, excitate 
    
class TFAdecoder13(tfattentiondecoder):
    def __init__(self, filter_num, latent_size):
        super().__init__(filter_num , latent_size)

    def forward(self, x, excitate):
        x = self.convt2( self.convt1(x) )
        x = torch.div( self.convt4(self.convt3( torch.div(x, excitate[1]))), excitate[0] )
        return x 
#*****************************12******************************************
class TFAencoder12(tfattentionencoder):
    def __init__(self, filter_num, latent_size, detach):
        super().__init__(filter_num, latent_size, detach)
        self.tfsequeeze1 = TFattention(insize = [21, 40])
        self.tfsequeeze2 = TFattention(insize = [19, 19])

    def forward(self, x):
        excitate1 = self.tfsequeeze1(x)
        z1 = self.conv1(torch.mul(x, excitate1))
        excitate2 = self.tfsequeeze2(z1).repeat(1, self.filter_num, 1, 1)
        z1 = torch.mul(z1, excitate2)
        z1 = self.conv2(z1)
        z2 = self.conv3( z1 )
        z2 = self.conv4(z2)
        z2 = z2.view(-1, 2, self.latent_size)
        excitate = [excitate1.detach(), excitate2.detach()] if self.detach == True else [excitate1, excitate2]
        return z1, z2, excitate 
    
class TFAdecoder12(tfattentiondecoder):
    def __init__(self, filter_num, latent_size):
        super().__init__(filter_num , latent_size)

    def forward(self, x, excitate):
        x = self.convt2( self.convt1(x) )
        x = torch.div( self.convt4(torch.div(self.convt3(x), excitate[1])), excitate[0] )
        return x 
#*****************************34******************************************
class TFAencoder34(tfattentionencoder):
    def __init__(self, filter_num, latent_size, detach):
        super().__init__(filter_num, latent_size, detach)
        self.tfsequeeze1 = TFattention(insize = [9, 9])
        self.tfsequeeze2 = TFattention(insize = [4, 4])

    def forward(self, x):
        z1 = self.conv1(x)
        z1 = self.conv2(z1)
        excitate1 = self.tfsequeeze1(z1)
        z1 = torch.mul(z1, excitate1.repeat(1, self.filter_num * 2, 1, 1 ) )
        z2 = self.conv3(z1)
        excitate2 = self.tfsequeeze2(z2)
        z2 = torch.mul(z2, excitate2.repeat(1, self.filter_num * 4, 1, 1) )
        z2 = self.conv4(z2)
        z2 = z2.view(-1, 2, self.latent_size)
        excitate = [excitate1.detach(), excitate2.detach()] if self.detach == True else [excitate1, excitate2]
        return z1, z2, excitate 
    
class TFAdecoder34(tfattentiondecoder):
    def __init__(self, filter_num, latent_size):
        super().__init__(filter_num , latent_size)

    def forward(self, x, excitate):
        x = torch.div( self.convt2(torch.div(self.convt1(x), excitate[1])), excitate[0] )
        x = self.convt4(self.convt3(x))
        return x 
