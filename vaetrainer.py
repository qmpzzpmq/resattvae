import torch.nn as nn
import torch
from tqdm import tqdm
from tensorboardX import SummaryWriter
from math import floor
from pdb import set_trace
from vaeextension import reporter
#from parallel import DataParallelModel, DataParallelCriterion

class mape(nn.Module):
    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction
    def forward(self, x, y):
        ret = torch.div(torch.abs(x-y), torch.abs(y) + torch.tensor(1e-15) )
        return torch.mean(ret) if self.reduction == "mean" else torch.sum(ret)

class vaeloss(nn.Module):
    def __init__(self, bcef, reduction, kldratio = 1):
        super().__init__()
        self.bcef = bcef
        self.reduction = reduction
        self.kldratio = kldratio
                            
    def forward(self, x_pred, x_true, mu, logvar):
        bce = self.bcef(x_pred, x_true)
        if self.reduction == 'sum':
            kld = -0.5 *  torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) * self.kldratio
        else:#mean
            kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp()) * self.kldratio
        return (bce + kld), bce, kld

class H5batchDataset(torch.utils.data.Dataset):
        def __init__(self, data_tensor, batch_size):
            self.data = data_tensor
            self.batch_size = batch_size
            self.len = floor(self.data.shape[0]/self.batch_size)

        def __getitem__(self, index): 
            x = torch.from_numpy(self.data[index*self.batch_size:(index+1)*self.batch_size,:]).float()
            x = (x+4.88)/7
            return x
                                            
        def __len__(self):
            return self.len
    
class vaetrainer(object): 
    def __init__(self, vae, opt, lr, device, device_ids, traindata, testdata, batch_size, log_dir, reduction = "mean"):
        assert opt == "adam", "current only adam"
        self.vae = vae
        self.opts = [torch.optim.Adam(vae.encoder.parameters(), lr = lr), torch.optim.Adam(vae.decoder.parameters(), lr = lr)]
        self.loss_fn = vaeloss(nn.MSELoss(reduction = reduction),reduction = reduction)
        self.metric_fn = mape(reduction=reduction)
        self.device = device
        self.device_ids = device_ids
        if len(device_ids) > 1:
            self.vae = vae.cuda(self.device_ids[0])
            self.vae = nn.DataParallel(self.vae, device_ids = device_ids, output_device = device_ids[0])
        traindata = H5batchDataset(traindata, batch_size)
        testdata = H5batchDataset(testdata, batch_size)
        self.traindata = torch.utils.data.DataLoader(traindata, batch_size= 1,  pin_memory = True, shuffle = True, num_workers = 1)
        self.testdata = torch.utils.data.DataLoader(testdata, batch_size= 1,  pin_memory = True, shuffle = True, num_workers = 1)
        self.trainlen = len(self.traindata)
        self.testlen = len(self.testdata)
        #extension
        self.reporter = reporter(log_dir=log_dir)
    def train(self, epoch):
        bias = (epoch - 1) * self.trainlen
        self.vae.train()
        for step, x_true in enumerate(tqdm(self.traindata)):
            if step % 10 == 0:
                torch.cuda.empty_cache()
            x_true = x_true.view(-1,1,21,40).to(self.device)
            for opt in self.opts:
                opt.zero_grad()
            x_pred, _, mu, logvar = self.vae(x_true)
            loss, bce, kld = self.loss_fn(x_pred, x_true, mu, logvar)
            loss.backward()
            for opt in self.opts:
                opt.step()
            self.reporter.trainstep(loss.item(), bce.item(), kld.item(), bias + step)
        self.reporter.trainepoch(epoch, self.trainlen)
        torch.cuda.empty_cache()
    def test(self,epoch):
        bias = (epoch - 1) * self.testlen
        self.vae.eval()
        for step, x_true in enumerate(tqdm(self.testdata)):
            if step % 10 == 0:
                torch.cuda.empty_cache()
            x_true = x_true.view(-1,1,21,40).to(self.device)
            x_pred, _, mu, logvar = self.vae(x_true)
            mape = self.metric_fn(x_pred, x_true)
            self.reporter.teststep(mape.item(), bias + step)
        emape = self.reporter.testepoch(epoch, self.testlen)
        self.reporter.imagecompare(x_pred, x_true, epoch)
        torch.cuda.empty_cache()
        return emape
