import argparse
import time
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils_functions import *
from my_models import ResGenerator as G
from my_models import GatedResGenerator as G1
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage import io as img
from skimage import transform
import warnings
warnings.filterwarnings("ignore")

seed = 2020
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
print('seed:'+str(seed))

def random_generate(Gs,Zs,NoiseWeight,reals,
                    start_scale,gen_num,scale_h,scale_w,outputdir,inputdir,part,masks_f=None,G_num_layer=3):
    if part=='f':
        images_curr = []
        n = 0
        fpadder = make_padder(1+2+G_num_layer*2)

        path = inputdir
        scales = len(Gs_f)
        

        # Gs = Gs[0:-1]


        for G_f,Z_f,noiseweight_f,real_f in zip(Gs,Zs,NoiseWeight,reals):

            g = G(3,256,3,1,0,3,0.01+1.7*(scales-n-1)/scales)
            ######  remember to change this factor
            g.load_state_dict(torch.load(path+'/'+str(n)+'/G_f.pth'))
            G_f = g
            G_f.to(torch.device("cuda:0"))
            del g
            G_f = G_f.eval()

            # import pdb;pdb.set_trace()
            print('scale=%s'%n)
            images_prev = images_curr
            images_curr = []
            h, w = real_f.shape[2], real_f.shape[3]
            h, w = int(h*scale_h), int(w*scale_w)
            real_f = F.interpolate(real_f,[h,w],mode='bilinear',align_corners=True)
            plt.imsave(outputdir+'/r.png', convert_image_np(real_f), vmin=0, vmax=1)
            real_f = real_f.squeeze(0).cpu()
            # G_f= G_f.to(torch.device("cuda:0"))
            for i in tqdm(range(gen_num)):
                data_aug = []
                if n==0:
                    z_curr = fpadder(Generate_noise([1,h,w],torch.device("cuda:0"),False,True))
                    # z_curr = generate_noise([1, Z_f.shape[2] - 2 * fpadder.padding[0], Z_f.shape[3] - 2 * fpadder.padding[0]], device=torch.device("cuda:0"))
                    # z_curr = z_curr.expand(1, 3, z_curr.shape[2], z_curr.shape[3])
                    # z_curr = Z_f
                else:
                    z_curr = fpadder(Generate_noise([3,h,w],torch.device("cuda:0"),False,False))
                    # z_curr = generate_noise([3, Z_f.shape[2] - 2 * fpadder.padding[0], Z_f.shape[3] - 2 * fpadder.padding[0]], device=torch.device("cuda:0"))
                # z_curr = fpadder(z_curr)
                if images_prev==[]:
                    I_prev = torch.zeros_like(z_curr).to(torch.device("cuda:0"))
                else:
                    I_prev = images_prev[i]
                    #SR?
                    # m = nn.Upsample([h,w],mode='bilinear',align_corners=True)
                    # I_prev = m(I_prev)
                    I_prev = F.interpolate(I_prev,[h,w],mode='bilinear',align_corners=True)
                    I_prev = fpadder(I_prev)
                #暂不可用！
                # if n<start_scale:
                #     z_curr = Z_f
                ###
                _augument = data_augmenter(real_f,torch.device("cuda:0"))
                # _augument = torch.zeros_like(real_f.unsqueeze(0).to(torch.device("cuda:0")))
                for j in range(G_num_layer):
                    _fpadder = make_padder(2+2*3+(G_num_layer-j)*2)
                    # data_aug.append(_fpadder(torch.ones_like(_augument).detach()))
                    # plt.imsave(outputdir+'/a.png', convert_image_np(_augument.detach()), vmin=0, vmax=1)
                    data_aug.append(_fpadder(_augument).detach())
                del _augument
                z_in = noiseweight_f * z_curr + I_prev
                # z_in = z_in.to(torch.device("cuda:0"))
                I_curr = G_f(z_in.detach(),data_aug,I_prev)
                images_curr.append(I_curr.detach())
                plt.imsave(outputdir+'/f_%s.png' %  (i), convert_image_np(I_curr.detach()), vmin=0, vmax=1)
            del G_f
            n+=1

    elif part=='b':
        images_curr = []
        n = 0
        G_num_layer = 3
        # bpadder = make_padder(1+2+G_num_layer*2)
        bpadder = make_padder(0)
        path = inputdir
        scales = len(Gs_b)
        
        # Gs = Gs[0:-1]


        for G_b,Z_b,noiseweight_b,real_b,mask_f in zip(Gs,Zs,NoiseWeight,reals,masks_f):
            print('scale=%s'%n)

            g = G1(3,256,3,1,0,3)
            g.load_state_dict(torch.load(path+'/'+str(n)+'/G_b.pth'))
            G_b = g
            G_b.to(torch.device("cuda:1"))
            del g
            G_b = G_b.eval()

            mask_f = mask_f.to(torch.device("cuda:1"))
            images_prev = images_curr
            images_curr = []
            h, w = real_b.shape[2], real_b.shape[3]
            h, w = int(h*scale_h), int(w*scale_w)
            real_b = F.interpolate(real_b,[h,w],mode='bilinear',align_corners=True)
            real_b = real_b.squeeze(0).cpu()
            for i in tqdm(range(gen_num)):
                if n==0:
                    z_curr = bpadder(Generate_noise([1,h,w],torch.device("cuda:1"),False,True))
                    # z_curr = Z_f
                else:
                    z_curr = bpadder(Generate_noise([3,h,w],torch.device("cuda:1"),False,False))
                if images_prev==[]:
                    I_prev = bpadder(torch.zeros_like(z_curr).to(torch.device("cuda:1")))
                else:
                    I_prev = images_prev[i]
                    #SR?
                    # m = nn.Upsample([z_curr.shape[2],z_curr.shape[3]],mode='bilinear',align_corners=True)
                    # I_prev = m(I_prev)
                    I_prev = F.interpolate(I_prev,[h,w],mode='bilinear',align_corners=True)
                # not available for now
                if n < start_scale:
                    z_curr = Z_b
                ###
                z_in = noiseweight_b * z_curr + I_prev
                I_curr = G_b(z_in.detach(),I_prev)
                images_curr.append(I_curr.detach())
                mask = torch.ones_like(mask_f) - mask_f
                I_curr = I_curr*mask - mask_f
                plt.imsave(outputdir+'/b_%s.png' %  (i), convert_image_np(I_curr.detach()), vmin=0, vmax=1)
            del G_b
            n+=1

def SR():
    pass
def harmonization():
    pass
def edit():
    pass
def paint2image():
    pass

def animation(Gs_f,Zs_f,NoiseWeight_f,reals_f,
            Gs_b,Zs_b,NoiseWeight_b,reals_b,
            start_scale,outputdir):
    pass
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, required=True)
    parser.add_argument('--path', type=str, required=True)
    parser.add_argument('--part', type=str, required=True)
    parser.add_argument('--start_scale', type=int, default=0)
    #for random generate
    parser.add_argument('--gen_num', type=int, default=50)
    parser.add_argument('--scale_h', type=float, default=1)
    parser.add_argument('--scale_w', type=float, default=1)
    parser = parser.parse_args()

    local_time = time.strftime("%m%d_%H:%M:%S", time.localtime())
    local_time = local_time.replace(":", "_")
    outputdir = '/home/chenjinshu/singan/SinGAN-master/app_result/' + str(parser.mode) + '/' + local_time
    inputdir = '/home/chenjinshu/singan/SinGAN-master/' + str(parser.path)

    if parser.part=='f':
        Gs_f = torch.load('%s/Gs_f.pth'%inputdir)
        Zs_f = torch.load('%s/Zs_f.pth'%inputdir)
        NoiseWeight_f = torch.load('%s/noiseweight_f.pth'%inputdir)
    elif parser.part=='b':
        Gs_b = torch.load('%s/Gs_b.pth'%inputdir)
        Zs_b = torch.load('%s/Zs_b.pth'%inputdir)
        NoiseWeight_b = torch.load('%s/noiseweight_b.pth'%inputdir)
    reals_f = torch.load('%s/reals_f.pth'%inputdir)
    reals_b = torch.load('%s/reals_b.pth'%inputdir)
    masks_f = torch.load('%s/masks_f.pth'%inputdir)

    if not os.path.exists(outputdir):
        os.makedirs(outputdir)
    
    if parser.mode=='r':
        if parser.part=='f':
            random_generate(Gs_f,Zs_f,NoiseWeight_f,reals_f,
                            parser.start_scale,parser.gen_num,
                            parser.scale_h,parser.scale_w,outputdir,inputdir,parser.part)
        elif parser.part=='b':
            random_generate(
                            Gs_b,Zs_b,NoiseWeight_b,reals_b,
                            parser.start_scale,parser.gen_num,
                            parser.scale_h,parser.scale_w,outputdir,inputdir,parser.part,masks_f=masks_f)
    elif parser.mode=='s':
        pass
    elif parser.mode=='h':
        pass
    elif parser.mode=='e':
        pass
    elif parser.mode=='p':
        pass
    elif parser.mode=='a':
        # not available for now
        animation(Gs_f,Zs_f,NoiseWeight_f,reals_f,
                Gs_b,Zs_b,NoiseWeight_b,reals_b,
                parser.start_scale,outputdir)
    else:
        print('???')