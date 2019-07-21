import argparse
import sys
from pdb import set_trace
from resattvae import resattvae
    
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')
parser = argparse.ArgumentParser()
parser.add_argument('--tag', type=str, default="none", help="tag name")
#model setting
parser.add_argument('--model', type=str, default="tfattention", choices=["tfattention", "baseline"],help="model select")
parser.add_argument('--input_h', type=int, default=21, help="input hight corresponding time dimension")
parser.add_argument('--input_w', type=int, default=40, help="input width corresponding frequency dimension")
parser.add_argument('--filter_num', type=int, default=16, help="Autoencoder filter number control")
parser.add_argument('--latent_size', type=int, default=90, help="latent size")
parser.add_argument('--attentiontype', type=str, default="SE", choices=["SE", "self"],help="sequeeze-excitation or self attention")
parser.add_argument('--lv1', type=int, default=2, help="attention level1")
parser.add_argument('--lv2', type=int, default=3, help="attention level2")
parser.add_argument('--batch_size', type=int, default=int(2e4), help="batch size")
parser.add_argument('--detach', type=str2bool, nargs='?', const=False, help='Turn on or off detach')
#training setting
parser.add_argument('--start_epoch',type=int, default=1)
parser.add_argument('--epochs', type=int, default=100, help="epoch numbers")
parser.add_argument('--lr', type=float, default=1e-3, help="learning rate")
parser.add_argument('--load_model', type=str, default=None, help="Load Model")
parser.add_argument('--device_ids', type=list, default=[0,1], help="all available GPU")
parser.add_argument('--opt', default='adam', type=str, choices=['adadelta', 'adam', 'noam'], help='Optimizer')
parser.add_argument('--reduction', default='mean', type=str, choices=["mean", "sum"], help='loss reduction')
parser.add_argument('--seed', default=1, type=int)
#data
parser.add_argument('--h5path', type=str, default="/prosjekt/studenter/haoyut/Dataset/timit")

args = parser.parse_args(sys.argv[1:])
resattvae(args)
