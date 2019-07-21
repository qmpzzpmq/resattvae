import torch
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter
from pdb import set_trace
"""
class saveer(object):
    def load_checkpoint(model, checkpoint, optimizer, loadOptimizer):
        if checkpoint != 'No':
            print("loading checkpoint...")
            model_dict = model.state_dict()
            modelCheckpoint = torch.load(checkpoint)
            pretrained_dict = modelCheckpoint['state_dict']
            # 过滤操作
            new_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
            model_dict.update(new_dict)
            # 打印出来，更新了多少的参数
            print('Total : {}, update: {}'.format(len(pretrained_dict), len(new_dict)))
            model.load_state_dict(model_dict)
            print("loaded finished!")
            # 如果不需要更新优化器那么设置为false
            if loadOptimizer == True:
                optimizer.load_state_dict(modelCheckpoint['optimizer'])
                print('loaded! optimizer')
            else:
                print('not loaded optimizer')
        else:
            print('No checkpoint is included')
            return model, optimizer
    def save_checkpoint(model, epoch, loss, ):
        torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'loss': lossMIN, 'optimizer': optimizer.state_dict(),
                                                                       checkpoint_path + '/m-' + launchTimestamp + '-' + str("%.4f" % lossMIN) + '.pth.tar')
"""                                                                           
class reporter(object):
    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir)
        self.loss, self.bce, self.kld, self.mape = [0]*4

    def trainstep(self, loss, bce, kld, step):
        self.writer.add_scalars(f"train/loss", {"loss":loss,"bce":bce,"kld":kld}, step)
        self.loss += loss
        self.bce += bce
        self.kld += kld
       
    def teststep(self, mape, step):
        self.writer.add_scalars(f"test/metrics", {"mape":mape}, step)
        self.mape += mape

    def trainepoch(self, epoch, trainlen):
        eloss, ebce, ekld = self.loss/trainlen, self.bce/trainlen, self.kld/trainlen
        self.writer.add_scalars(f"summary/loss", {"loss": eloss,"bce": ebce,"kld": ekld}, epoch)
        self.loss, self.bce, self.kld = [0]*3
        return eloss, ebce, ekld

    def testepoch(self, epoch, testlen):
        emape = self.mape/testlen
        self.writer.add_scalars(f"summary/metrics", {"mape": emape}, epoch)
        self.mape = 0
        return emape

    def imagecompare(self, x_pred, x_true, epoch, n=8):
        comparison = torch.cat([x_pred[:n,:,:,:],x_true[:n,:,:,:]])
        comparison = make_grid(comparison, padding=5)
        self.writer.add_image('Reconstruction Image', comparison, epoch)
