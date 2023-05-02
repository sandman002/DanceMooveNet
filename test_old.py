import argparse
import math
import random
import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="2"

import numpy as np
import torch
from torch import nn, autograd, optim
from torch.nn import functional as F
from torch.utils import data
import torch.distributed as dist
from torchvision import transforms, utils
# from torchvision.io import write_video
from tqdm import tqdm
import torchvision
from torch.autograd import Variable
from keypointLoader import KeypointsDataset
from torch.utils.data import DataLoader

# from model_old_noise_classmixed import MoveNet
from model_old_noise_classmixed import MoveNet
torch.manual_seed(1111)

label_to_id = {'mBR0': 0, 'mBR1': 1, 'mBR2': 2, 'mBR3': 3, 'mBR4': 4, 'mBR5': 5, 
               'mPO0': 6, 'mPO1': 7, 'mPO2': 8, 'mPO3': 9, 'mPO4': 10, 'mPO5': 11, 
               'mLO0': 12, 'mLO1': 13, 'mLO2': 14, 'mLO3': 15, 'mLO4': 16, 'mLO5': 17, 
               'mMH0': 18, 'mMH1': 19, 'mMH2': 20, 'mMH3': 21, 'mMH4': 22, 'mMH5': 23, 
               'mLH0': 24, 'mLH1': 25, 'mLH2': 26, 'mLH3': 27, 'mLH4': 28, 'mLH5': 29, 
               'mHO0': 30, 'mHO1': 31, 'mHO2': 32, 'mHO3': 33, 'mHO4': 34, 'mHO5': 35, 
               'mWA0': 36, 'mWA1': 37, 'mWA2': 38, 'mWA3': 39, 'mWA4': 40, 'mWA5': 41, 
               'mKR0': 42, 'mKR1': 43, 'mKR2': 44, 'mKR3': 45, 'mKR4': 46, 'mKR5': 47, 
               'mJS0': 48, 'mJS1': 49, 'mJS2': 50, 'mJS3': 51, 'mJS4': 52, 'mJS5': 53, 
               'mJB0': 54, 'mJB1': 55, 'mJB2': 56, 'mJB3': 57, 'mJB4': 58, 'mJB5': 59}

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag

def d_logistic_loss(real_pred, fake_pred):
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)

    return real_loss.mean() + fake_loss.mean()

def g_nonsaturating_loss(fake_pred):
    loss = F.softplus(-fake_pred).mean()

    return loss

def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)



def generate(args, g_ema, device, label=0, noise=None, suffix='0'):
    
    with torch.no_grad():
        
        g_ema.eval()
        if noise == None:
            noise = torch.randn(args.batch_size, args.style_dim, device=device)
   
            
        gen_tau = (torch.cuda.FloatTensor(torch.randint(0, 1, (1,1), device=device).type(torch.cuda.FloatTensor)))
        class_labels = torch.tensor([label]).to(device)
       
        
        jj = 0
       
        lat_i = 0
        pose = []
        for t in tqdm(range(args.frames)):
            pose.append(g_ema(noise,class_labels,gen_tau+t,0.6).detach().cpu().numpy())

        pose_seq = np.array(pose)
        print(pose_seq.shape)
        os.makedirs(args.samplepath, exist_ok=True)
        np.save(args.samplepath+'/'+suffix,pose_seq)

         

if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser(description="StyleGAN2 trainer")

    parser.add_argument('--samplepath', type=str, default='./sample/', help='model architectures (stylegan2 | swagan)')
    parser.add_argument('--chkpath', type=str, default='./ckpt/', help='model architectures (stylegan2 | swagan)')
    parser.add_argument("--datapath", type=str, default='./dataset/AIST/annotation/keypoints3d/', help="pose pickle path")
    parser.add_argument('--batch_size', type=int, default=4, help='training batch size')
    parser.add_argument('--iter', type=int, default=100000, help='training iterations')
    parser.add_argument('--transit_start', type=int, default=2000, help='starting iteration of conditional transition')
    parser.add_argument('--transit_end', type=int, default=4000, help='ending iteration of conditional transition')
    parser.add_argument('--style_dim', type=int, default=128, help='ending iteration of conditional transition')
    parser.add_argument('--dyn_size', type=int, default=128, help='ending iteration of conditional transition')
    parser.add_argument('--embed', type=int, default=64, help='action embedding dim size')
    parser.add_argument('--num_keys', type=int, default=17, help='ending iteration of conditional transition')
    parser.add_argument('--num_class', type=int, default=60, help='number of action classes')
    parser.add_argument('--frames', type=int, default=300, help='number of frames')
    parser.add_argument('--label', type=int, default=0, help='dance label')
    parser.add_argument("--lr", type=float, default=0.002, help="learning rate")

    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="path to the checkpoints to resume training",
    )
    parser.add_argument(
        "--d_reg_every",
        type=int,
        default=16,
        help="interval of the applying r1 regularization",
    )
    parser.add_argument(
        "--g_reg_every",
        type=int,
        default=4,
        help="interval of the applying path length regularization",
    )


    args = parser.parse_args()


    g_ema = MoveNet(args.style_dim, args.embed, args.num_keys, args.num_class).to(device)
    g_ema.eval()



    if args.ckpt is not None:
        print("load model:", args.ckpt)
        ckpt = torch.load(args.ckpt, map_location=lambda storage, loc: storage)
        try:
            ckpt_name = os.path.basename(args.ckpt)
            args.start_iter = int(os.path.splitext(ckpt_name)[0])

        except ValueError:
            pass

        g_ema.load_state_dict(ckpt["g_ema"], strict=False)





    pose_pkl_path = args.datapath
    file_list = []
    label = args.label
    noise = torch.randn(args.batch_size, args.style_dim, device=device)
    base = int(os.path.basename(args.ckpt[:-3]))
    for rr in range(0,60):
        generate(args, g_ema, device, rr, noise, suffix=str(base).zfill(3)+"_"+str(rr).zfill(3))