import argparse
import math
import random
import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="1"

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

from model_old_noise_classmixed import MoveNet, MoveDiscriminate


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

def train(args, dataloader, gen, dis, g_optim, d_optim, g_ema, device):
    pbar = range(args.iter)
    if not os.path.exists(args.chkpath):
        os.makedirs(args.chkpath)
    if not os.path.exists(args.samplepath):
        os.makedirs(args.samplepath)

    pbar = tqdm(pbar, initial=args.start_iter, dynamic_ncols=True, smoothing=0.01)

    mean_path_length = 0
    lambda_transition = 0
    d_loss_val = 0
    d2_loss_val = 0
    r1_loss = torch.tensor(0.0, device=device)
    g_loss_val = 0
    path_loss = torch.tensor(0.0, device=device)
    path_lengths = torch.tensor(0.0, device=device)
    mean_path_length_avg = 0
    loss_dict = {}
    offset = [1,2,4,8, 16,32]
    kk = 0
    accum = 0.5 ** (32 / (10 * 1000))
    transition_denominator = 1/(args.transit_end - args.transit_start+1e-7)
    for it_id in pbar:
        i = it_id + args.start_iter

        if i > args.iter:
            print("Done!")
            break
        data = next(iter((dataloader)))
        k3d_optim = data[0]
       
        music_id_str = data[1]
        ran_start_frameid = data[2].unsqueeze(1).to(device).type(torch.cuda.FloatTensor)
        frame_len = data[3]

        music_id = torch.tensor([label_to_id[string_id] for string_id in music_id_str], dtype=torch.int64, device=device)

        frame_m1 = k3d_optim[:,0,:].to(device).type(torch.cuda.FloatTensor)
        frame_0 =  k3d_optim[:,1*offset[kk],:].to(device).type(torch.cuda.FloatTensor)
        frame_p1 = k3d_optim[:,2*offset[kk],:].to(device).type(torch.cuda.FloatTensor)


        requires_grad(gen, False)
        requires_grad(dis, True)
        gen_tau = torch.zeros(args.batch_size,1).to(device).type(torch.cuda.FloatTensor)
        for jj in range(args.batch_size):
            gen_tau[jj,0] = Variable(torch.cuda.FloatTensor(torch.randint(offset[kk], frame_len[jj]-offset[kk], (1,1), device=device).type(torch.cuda.FloatTensor)))
        


        zm1 = ((gen_tau ))
        z0  = ((gen_tau + 1*offset[kk]))
        zp1 = ((gen_tau + 2*offset[kk]))
        # print((gen_tau - offset[kk])/vid_span.unsqueeze(1), 'gen_tau')

        noise = torch.randn(args.batch_size, args.style_dim, device=device)

        fake_frame_m1 = gen(noise,music_id,zm1,lambda_transition)
        fake_frame_0  = gen(noise,music_id, z0,lambda_transition)
        fake_frame_p1 = gen(noise,music_id,zp1,lambda_transition)


        fake4dis = torch.cat([fake_frame_m1, fake_frame_0, fake_frame_p1], 0)
        fake_pred = dis(fake4dis, music_id, (gen_tau),  offset[kk], lambda_transition)

        read4dis = torch.cat([frame_m1, frame_0, frame_p1], 0)

        real_pred = dis(read4dis, music_id, ran_start_frameid, offset[kk], lambda_transition)

        d_loss = d_logistic_loss(real_pred, fake_pred)

        loss_dict["d"] = d_loss
        dis.zero_grad()
        d_loss.backward()
        d_optim.step()


        requires_grad(gen, True)
        requires_grad(dis, False)

        gen_tau = torch.zeros(args.batch_size,1).to(device).type(torch.cuda.FloatTensor)
        for jj in range(args.batch_size):
            gen_tau[jj,0] = Variable(torch.cuda.FloatTensor(torch.randint(offset[kk], frame_len[jj]-offset[kk], (1,1), device=device).type(torch.cuda.FloatTensor)))
        


        zm1 = ((gen_tau ))
        z0  = ((gen_tau + 1*offset[kk]))
        zp1 = ((gen_tau + 2*offset[kk]))
        # print((gen_tau - offset[kk])/vid_span.unsqueeze(1), 'gen_tau')

        noise = torch.randn(args.batch_size, args.style_dim, device=device)

        fake_frame_m1 = gen(noise,music_id,zm1,lambda_transition)
        fake_frame_0  = gen(noise,music_id, z0,lambda_transition)
        fake_frame_p1 = gen(noise,music_id,zp1,lambda_transition)

        fake4dis = torch.cat([fake_frame_m1, fake_frame_0, fake_frame_p1], 0)
        fake_pred = dis(fake4dis, music_id, (gen_tau), offset[kk], lambda_transition)

        g_loss = g_nonsaturating_loss(fake_pred) 

        loss_dict["g"] = g_loss

        gen.zero_grad()
        g_loss.backward()
        g_optim.step()

        accumulate(g_ema, g_module, accum)
        d_loss_val = loss_dict["d"].mean().item()
        g_loss_val = loss_dict["g"].mean().item()
       

        pbar.set_description(
            (
                f"d: {d_loss_val:.4f}; g: {g_loss_val:.4f}; t: {lambda_transition:.4f}"
            )
        )



        if i >= args.transit_start: #begin class infulence on style
            factor = (i - args.transit_start)/(args.transit_end - args.transit_start+1e-7)
            lambda_transition = np.clip(factor, a_min=0, a_max=1)

        if(i%500==0):
                kk = (kk+1)%6 #offset index

        if i % 10000 == 0:
            torch.save(
                {
                    "g": g_module.state_dict(),
                    "d": d_module.state_dict(),
                    "g_ema": g_ema.state_dict(),
                    "g_optim": g_optim.state_dict(),
                    "d_optim": d_optim.state_dict(),
                    "args": args,
                },
                args.chkpath+f"/{str(i).zfill(6)}.pt",
            )

        
    torch.save(
        {
            "g": g_module.state_dict(),
            "d": d_module.state_dict(),
            "g_ema": g_ema.state_dict(),
            "g_optim": g_optim.state_dict(),
            "d_optim": d_optim.state_dict(),
            "args": args,
        },
        args.chkpath+f"/{str(i).zfill(6)}.pt",
    )



if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser(description="StyleGAN2 trainer")

    parser.add_argument('--samplepath', type=str, default='./sample/', help='model architectures (stylegan2 | swagan)')
    parser.add_argument('--chkpath', type=str, default='./ckpt/', help='model architectures (stylegan2 | swagan)')
    parser.add_argument("--datapath", type=str, default='./dataset/AIST/annotation/keypoints3d/', help="pose pickle path")
    parser.add_argument('--batch_size', type=int, default=4, help='training batch size')
    parser.add_argument('--iter', type=int, default=100000, help='training iterations')
    parser.add_argument('--transit_start', type=int, default=12000, help='starting iteration of conditional transition')
    parser.add_argument('--transit_end', type=int, default=14000, help='ending iteration of conditional transition')
    parser.add_argument('--style_dim', type=int, default=128, help='ending iteration of conditional transition')
    parser.add_argument('--dyn_size', type=int, default=128, help='ending iteration of conditional transition')
    parser.add_argument('--embed', type=int, default=128, help='action embedding dim size')
    parser.add_argument('--num_keys', type=int, default=17, help='ending iteration of conditional transition')
    parser.add_argument('--num_class', type=int, default=60, help='number of action classes')
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

    gen = MoveNet(args.style_dim, args.embed, args.num_keys, args.num_class).to(device)
    dis = MoveDiscriminate(args.style_dim, args.embed, args.num_keys, args.num_class).to(device)


    g_reg_ratio = args.g_reg_every / (args.g_reg_every + 1)
    d_reg_ratio = args.d_reg_every / (args.d_reg_every + 1)

    g_optim = optim.Adam(
        gen.parameters(),
        lr=args.lr * g_reg_ratio,
        betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio),
    )
    d_optim = optim.Adam(
        dis.parameters(),
        lr=args.lr * d_reg_ratio,
        betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio),
    )

    g_ema = MoveNet(args.style_dim, args.embed, args.num_keys, args.num_class).to(device)
    g_ema.eval()
    accumulate(g_ema, gen, 0)
    args.start_iter = 0


    g_module = gen
    d_module = dis
    

    if args.ckpt is not None:
        print("load model:", args.ckpt)
        ckpt = torch.load(args.ckpt, map_location=lambda storage, loc: storage)
        try:
            ckpt_name = os.path.basename(args.ckpt)
            args.start_iter = int(os.path.splitext(ckpt_name)[0])

        except ValueError:
            pass

        gen.load_state_dict(ckpt["g"], strict=False)

        # discriminator.load_state_dict(ckpt["d"])
        dis.load_state_dict(ckpt["d"], strict=False)
        g_ema.load_state_dict(ckpt["g_ema"], strict=False)

        g_optim.load_state_dict(ckpt["g_optim"])
        d_optim.load_state_dict(ckpt["d_optim"])




    pose_pkl_path = args.datapath
    file_list = []

    for file_name in os.listdir(pose_pkl_path):
        if file_name.endswith('.pkl'):
            file_list.append(os.path.join(pose_pkl_path, file_name))


    dataset = KeypointsDataset(file_list, max_len=68)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    train(args, dataloader, gen, dis, g_optim, d_optim, g_ema, device)