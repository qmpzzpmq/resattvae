import sys
import time
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
from torchvision.utils import save_image
import h5py
from pdb import set_trace
from torchvision.utils import make_grid
from tqdm import tqdm

import vaetrainer as trainer

def resattvae(args):
    torch.manual_seed(args.seed)

    #Initialization
    if len(args.device_ids) == 0:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:{}'.format(args.device_ids[0]))
        #torch.cuda.set_device(args.device_ids[0])
        args.batch_size = args.batch_size * len(args.device_ids)
    
    #Data setting
    fbnktrn, fbnktst  = h5py.File(os.path.join(args.h5path,"trainFbank840.h5")), h5py.File(os.path.join(args.h5path,"testFbank840.h5"))
    TrnSet, TstSet = fbnktrn['timit'], fbnktst['timit']
    
    #Model
    if args.model == "baseline":
        import baselineautoencoder as model
        encoder = model.encoder(args.filter_num, args.latent_size)
        decoder = model.decoder(args.filter_num, args.latent_size)
        vae = model.vae(encoder = encoder, decoder = decoder)
    else:
        if args.attentiontype == "self": 
            import tfattentionautoencoder2 as model
        else:
            import tfattentionautoencoder as model
        if (args.lv1 == 2) & (args.lv2 == 3):
            encoder = model.TFAencoder23(args.filter_num, args.latent_size, detach=args.detach)
            decoder = model.TFAdecoder23(args.filter_num, args.latent_size)
        elif (args.lv1 == 2) & (args.lv2 == 4):
            encoder = model.TFAencoder24(args.filter_num, args.latent_size, detach=args.detach)
            decoder = model.TFAdecoder24(args.filter_num, args.latent_size)
        elif (args.lv1 == 1) & (args.lv2 == 4):
            encoder = model.TFAencoder14(args.filter_num, args.latent_size, detach=args.detach)
            decoder = model.TFAdecoder14(args.filter_num, args.latent_size)
        elif (args.lv1 == 1) & (args.lv2 == 3):
            encoder = model.TFAencoder13(args.filter_num, args.latent_size, detach=args.detach)
            decoder = model.TFAdecoder13(args.filter_num, args.latent_size)
        elif (args.lv1 == 1) & (args.lv2 == 2):
            encoder = model.TFAencoder12(args.filter_num, args.latent_size, detach=args.detach)
            decoder = model.TFAdecoder12(args.filter_num, args.latent_size)
        elif (args.lv1 == 3) & (args.lv2 == 4):
            encoder = model.TFAencoder34(args.filter_num, args.latent_size, detach=args.detach)
            decoder = model.TFAdecoder34(args.filter_num, args.latent_size)
        vae = model.tfattentionvae(encoder = encoder, decoder = decoder)

    #Trainning setting
    vaetrainer = trainer.vaetrainer(vae, args.opt, args.lr, device, args.device_ids, TrnSet, TstSet, args.batch_size,\
            "tensorboard/"+args.tag, reduction=args.reduction)

    for epoch in range(args.start_epoch, args.start_epoch + args.epochs):
        print(f"Training {epoch} ({args.start_epoch}->{args.start_epoch + args.epochs-1})")
        vaetrainer.train(epoch)
        emape = vaetrainer.test(epoch)
        print(f"Mape Epoch{epoch}: {emape}")
