#!/usr/bin/env python
# coding: utf-8

# import torch.nn.functional as F
# import torch.optim as optim
# from apex import amp
# from tqdm import tqdm
# from skimage import io as img
# import torchvision.models as models
# from torchsummaryX import summary as modelsummary
# import nni

import time
import os
import warnings
import shutil
import argparse
import logging
import math
import torch
import torch.nn as nn
import numpy as np
import scipy.misc
import matplotlib
import matplotlib.pyplot as plt
import colorama
from utils_functions import ran_gen, seconds2time, get_logger, data_augmenter, plot_sinloss, make_padder, get_reals, Generate_noise, convert_image_np, draw_concat, calc_gradient_penalty, reset_grads, modify_scales, get_seg, calc_local_rec
from training_functions import organise_models, recon_loss, VGGPerceptualLoss
from configs import get_configs, LogbookFormatter
from progress_bar import create_progressbar
from evaluation import calculate_sifid_given_paths, calculate_cs
warnings.filterwarnings("ignore")
matplotlib.use('Agg')
# np.set_printoptions(threshold = np.inf)
# @profile
def main():

    # params = nni.get_next_parameter()

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, required=True)
    parser.add_argument('--config_file', type=str, required=True)
    parser.add_argument('--log', type=bool, default=True)
    parser.add_argument('--logbook', type=str, default='log.txt')
    logger = get_logger()
    parser = get_configs(parser, logger)
    opts = parser.parse_args()

    red = colorama.Fore.RED
    green = colorama.Fore.GREEN
    white = colorama.Fore.WHITE
    cyan = colorama.Fore.CYAN
    reset = colorama.Style.RESET_ALL
    bright = colorama.Style.BRIGHT
    dim = colorama.Style.DIM

    seed = opts.seed
    # seed = params['seed']
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    img_path = opts.img_input_dir
    anot_path = opts.anot_input_dir
    output_dir = opts.output_dir

    local_time = time.strftime("%m%d_%H%M%S", time.localtime())
    dir2save = '%s/%s_seed%s/' % (output_dir, local_time, seed)
    gen_num = opts.gen_num
    dir2gen = '%sran_gen/' % (dir2save)

    try:
        os.makedirs(dir2save)
        os.makedirs(dir2gen)
        shutil.copyfile('%s/%s' % (os.getcwd(), opts.config_file), '%s/%s' % (dir2save, opts.config_file))
        shutil.copyfile('%s/my_models.py' % os.getcwd(), '%s/my_models.py' % dir2save)
        shutil.copyfile('%s/run.py' % os.getcwd(), '%s/run.py' % dir2save)
        shutil.copyfile('%s/training_functions.py' % os.getcwd(), '%s/training_functions.py' % dir2save)
        shutil.copyfile('%s/utils_functions.py' % os.getcwd(), '%s/utils_functions.py' % dir2save)
    except OSError:
        raise Exception("Files ERROR!")
    if opts.log:
        logbook = opts.logbook
        logpath = dir2save + logbook
        loghandler = logging.FileHandler(filename=logpath, mode="a", encoding="utf-8")
        loghandler.setLevel(logging.INFO)
        logbook_formatter = LogbookFormatter(fmt="[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        loghandler.setFormatter(logbook_formatter)
        logger.addHandler(loghandler)

    ### can't Loading Weights at present
    weights2load = 0
    G_weights2load = ''
    D_weights2load = ''
    ###

    Gs = []
    Zs = []
    Ds = []
    reals = []
    masks = []
    noises = []
    NoiseWeight = []

    errD2plot = []
    errG2plot = []

    mode = opts.mode
    channels = opts.channels
    kernel_size = opts.kernel_size
    stride = opts.stride

    if_padding = opts.if_padding
    if_lazy = opts.if_lazy

    G_num_layer = opts.G_num_layer
    D_num_layer = opts.D_num_layer
    if mode == 'f':
        weight4style = opts.weight4style
    scale_base = opts.scale_base
    # scales = opts.scales
    scales = modify_scales(anot_path, scale_base)
    logger.info('-' * 80)
    logger.info(green + '[INFO]: scales are set to %s' % scales + reset)

    out_channels = opts.out_channels

    lr_g = opts.lr_g
    lr_d = opts.lr_d

    iters_list = [int(i) for i in opts.iters_list]
    D_steps = opts.D_steps
    G_steps = opts.G_steps
    lambda_grad = opts.lambda_grad
    n_segments = opts.n_segments
    compactness = opts.compactness
    sigma = opts.sigma
    start_label = opts.start_label

    device = torch.device("cuda:0")

    alpha4rec_ini = opts.alpha4rec
    alpha4cos_ini = opts.alpha4cos
    alpha4vgg_ini = opts.alpha4vgg
    p_loss = VGGPerceptualLoss(resize=False, device=device) if alpha4vgg_ini != 0 else 0
    ###
    # factor4rec = calc_factor('rec', scales)
    # factor4cos = calc_factor('cos', scales)
    # factor4vgg = calc_factor('vgg', scales)
    ###

    noise_weight = opts.noise_weight
    noise_weight_ini = noise_weight

    p4flip = opts.p4flip

    torch.backends.cudnn.benchmark = True
    # amp.register_float_function(torch, 'sigmoid')

    reals, masks = get_reals(mode, img_path, anot_path, scales, scale_base, reals, channels, masks)
    reals, masks = reals[::-1], masks[::-1]

    reals_b, reals_fa, masks_b, masks_f = [], [], [], []
    for _ in reals:
        reals_b.append(_[1])
        reals_fa.append(_[3])
    for _ in masks:
        masks_b.append(_[0])
        masks_f.append(_[1])

    for scale_num in range(scales):
        outfile_dir = '%s%s/' % (dir2save, scale_num)
        try:
            os.makedirs(outfile_dir)
        except OSError:
            raise Exception("Files ERROR!")
        _, __, ___, ____, _____ = reals[scale_num][0], reals[scale_num][1], reals[scale_num][2], reals[scale_num][3], masks_f[scale_num]
        plt.imsave('%s/real_original.png' %  (outfile_dir), convert_image_np(_), vmin=0, vmax=1)
        plt.imsave('%s/real_background.png' %  (outfile_dir), convert_image_np(__), vmin=0, vmax=1)
        plt.imsave('%s/real_foregrounds.png' %  (outfile_dir), convert_image_np(___), vmin=0, vmax=1)
        plt.imsave('%s/real_foreground_a.png' %  (outfile_dir), convert_image_np(____), vmin=0, vmax=1)
        scipy.misc.toimage(convert_image_np(_____[:, 0, :, :][None, :, :, :])).save('%s/mask_f.png' %  (outfile_dir))
    torch.save(reals_fa, dir2save+'reals_f.pth')
    torch.save(reals_b, dir2save+'reals_b.pth')
    torch.save(masks_f, dir2save+'masks_f.pth')
    logger.info('-' * 80)
    logger.info(green + '[INFO]: data prepared!' + reset)
    logger.info('-' * 80)
    torch.cuda.synchronize()
    start_time = time.time()
    logger.info(green + '[INFO]: training starts at %s' % time.strftime("%H:%M:%S", time.localtime()) + reset)
    logger.info('-' * 80)

    for scale_num in range(scales):

        iters = iters_list[scale_num]
        outfile_dir = '%s%s/' % (dir2save, scale_num)
        real_curr = reals[scale_num]
        x = np.random.choice(iters, int(iters*p4flip), replace=False)
        # real_seg = get_seg(real_curr[3], n_segments=n_segments, compactness=compactness, sigma=sigma, start_label=start_label)

        zeros = torch.zeros_like(real_curr[3]).to(device)
        edge_w, edge_h = math.ceil(0.1*real_curr[3].shape[3]), math.ceil(0.1*real_curr[3].shape[2])
        for i in range(edge_w):
            zeros[:,:,:,i] = 1.
        for i in range(real_curr[3].shape[3]-edge_w, real_curr[3].shape[3]):
            zeros[:,:,:,i] = 1.
        for i in range(edge_h):
            zeros[:,:,i,:] = 1.
        for i in range(real_curr[3].shape[2]-edge_h, real_curr[3].shape[2]):
            zeros[:,:,i,:] = 1.
        assert zeros[0,0,0,0] == 1

        if mode == 'f':
            alpha4cos = alpha4cos_ini
            if scale_num >= scales: # 4 5
                alpha4rec = alpha4rec_ini * 10
            else: # 0 1 2 3
                alpha4rec = alpha4rec_ini
            real_curr[3] = real_curr[3].to(device)
            h, w = real_curr[3].shape[2], real_curr[3].shape[3]
            D, G, optimizerD, optimizerG, schedulerD, schedulerG = organise_models(
                mode, device, weights2load, lr_g, lr_d, channels, kernel_size, stride, if_padding,
                G_num_layer, D_num_layer, out_channels, factor=0.01+weight4style*(scales-scale_num-1)/scales
                )
        elif mode == 'b':
            # if scale_num <= 0:
            #     lr_g = 0.0001
            #     lr_d = 0.0001
            alpha4rec = alpha4rec_ini
            alpha4cos = alpha4cos_ini
            real_curr[1] = real_curr[1].to(device)
            h, w = real_curr[1].shape[2], real_curr[1].shape[3]
            D, G, optimizerD, optimizerG, schedulerD, schedulerG = organise_models(
                mode, device, weights2load, lr_g, lr_d, channels, kernel_size, stride, if_padding, G_num_layer, D_num_layer, out_channels
                )

        # [D, G], [optimizerD, optimizerG] = amp.initialize([D, G], [optimizerD, optimizerG], opt_level='O1', num_losses=14)

        # p_loss = 0
        #p_loss = p_loss.to(device)
        r_loss = recon_loss(False)
        r_loss = r_loss.to(device)

        if if_padding:
            padder = make_padder(0)
        else:
            padder = make_padder((G_num_layer-1)*1+2+1)
        # if opts.ani==True:
        #     fpadder = make_padder(0)
        #     h_f = h_f + (1+2+G_num_layer*2)*2
        #     w_f = w_f + (1+2+G_num_layer*2)*2
        noise_1 = padder(Generate_noise([channels, h, w], device=device, if_0=True, if_c_same=False))
        epoch_iterator = create_progressbar(
            iterable=range(iters),
            desc="Training scale [{}/{}]".format(scale_num, scales-1),
            offset=0, leave=True, logging_on_update=False, logging_on_close=True, postfix=True
        )
        for i in epoch_iterator:
            epoch_iterator.set_description('Scale [{}/{}], Iteration [{}/{}]'.format(scale_num+1, scales, i+1, iters))
            if mode == 'f':
                if i >= 1600 and scale_num > 0:
                    alpha4rec = alpha4rec_ini
                styles_ref = []
                _tmp = real_curr[3].squeeze(0).cpu()
                for cnt in range(G_num_layer*2+2):
                    if if_padding:
                        _padder = make_padder(0)
                    else:
                        _padder = make_padder(2+2*3+(G_num_layer-1-cnt)*1)
                    _augment = data_augmenter(_tmp, device=device)
                    _augment_ = _padder(_augment)
                    styles_ref.append(_augment_.detach())
                del _augment, _augment_

            if Gs == []:
                noise_1 = padder(Generate_noise([1, h, w], device=device, if_0=False, if_c_same=True))
                noise_2 = padder(Generate_noise([1, h, w], device=device, if_0=False, if_c_same=True))
                # noise_2_f = padder(get_slerp_interp([1, h_f, w_f], device=device, iters=iters, iter_curr=i, if_c_same=True, start=noise_2_f_s, end=noise_2_f_e))
            else:
                noise_2 = padder(Generate_noise([channels, h, w], device=device, if_0=False, if_c_same=False))
                # noise_2_f = padder(get_slerp_interp([channels, h_f, w_f], device=device, iters=iters, iter_curr=i, if_c_same=False, start=noise_2_f_s, end=noise_2_f_e))

            for j in range(D_steps):
                if (j == 0) & (i == 0):
                    if Gs == []:
                        noise_3 = padder(Generate_noise([channels, h, w], device=device, if_0=True, if_c_same=False))
                        prev = torch.full([1, channels, h, w], 0, device=device)
                        _ = prev
                        prev = padder(prev)
                        noise_weight = 1
                    else:
                        criterion = nn.MSELoss()
                        if mode == 'f':
                            prev = padder(draw_concat(Gs, Zs, reals_fa, NoiseWeight, _, 'rand', kernel_size, channels, device, padder, G_num_layer, mode))
                            noise_3 = draw_concat(Gs, Zs, reals_fa, NoiseWeight, _, 'rec', kernel_size, channels, device, padder, G_num_layer, mode)
                            RMSE = torch.sqrt(criterion(real_curr[3], noise_3))
                        elif mode == 'b':
                            prev = padder(draw_concat(Gs, Zs, reals_b, NoiseWeight, _, 'rand', kernel_size, channels, device, padder, G_num_layer, mode))
                            noise_3 = draw_concat(Gs, Zs, reals_b, NoiseWeight, _, 'rec', kernel_size, channels, device, padder, G_num_layer, mode)
                            RMSE = torch.sqrt(criterion(real_curr[1], noise_3))
                        noise_weight = noise_weight_ini*RMSE
                        noise_3 = padder(noise_3)
                else:
                    if mode == 'f':
                        prev = padder(draw_concat(Gs, Zs, reals_fa, NoiseWeight, _, 'rand', kernel_size, channels, device, padder, G_num_layer, mode))
                    elif mode == 'b':
                        prev = padder(draw_concat(Gs, Zs, reals_b, NoiseWeight, _, 'rand', kernel_size, channels, device, padder, G_num_layer, mode))
                if Gs == []:
                    noise = noise_2
                else:
                    noise = noise_weight * noise_2 + prev
                D.zero_grad()
                if mode == 'f':
                    output = D(real_curr[3])
                elif mode == 'b':
                    output = D(real_curr[1])
                if i in x:
                    errD_real = output.mean()
                else:
                    errD_real = -output.mean()
                # with amp.scale_loss(errD_real, optimizerD, loss_id=0) as errD_real:
                #     errD_real.backward(retain_graph=True)
                errD_real.backward(retain_graph=True)
                if i in x:
                    errD_real = -errD_real
                if mode == 'f':
                    fake = G(noise.detach(), styles_ref, prev)
                elif mode == 'b':
                    fake = G(noise.detach(), prev)
                output = D(fake.detach())
                if i in x:
                    errD_fake = -output.mean()
                else:
                    errD_fake = output.mean()
                # with amp.scale_loss(errD_fake, optimizerD, loss_id=1) as errD_fake:
                #     errD_fake.backward(retain_graph=True)
                errD_fake.backward(retain_graph=True)
                if i in x:
                    errD_fake = -errD_fake

                if mode == 'f':
                    gradient_penalty = calc_gradient_penalty(D, real_curr[3], fake, lambda_grad, device)
                elif mode == 'b':
                    gradient_penalty = calc_gradient_penalty(D, real_curr[1], fake, lambda_grad, device)
                # with amp.scale_loss(gradient_penalty, optimizerD, loss_id=2) as gradient_penalty:
                #     gradient_penalty.backward()
                gradient_penalty.backward()

                optimizerD.step()
                D.zero_grad()
                optimizerD.zero_grad()

            _errD_real = errD_real.item()
            _errD_fake = errD_fake.item()
            _gradient_penalty = gradient_penalty.item()
            del errD_real, errD_fake, gradient_penalty
            _errD = _errD_real + _errD_fake + _gradient_penalty
            errD2plot.append([_errD_real, _errD_fake, _gradient_penalty])
            schedulerD.step(_errD)
            for j in range(G_steps):
                G.zero_grad()
                ###
                output = D(fake)
                ###
                errG = -output.mean()
                # with amp.scale_loss(errG, optimizerG, loss_id=3) as errG:
                #     errG.backward(retain_graph=True)
                errG.backward(retain_graph=True)
                Z_opt = noise_weight * noise_1 + noise_3
                if mode == 'f':
                    _tmp = G(Z_opt.detach(), styles_ref, noise_3)
                elif mode == 'b':
                    _tmp = G(Z_opt.detach(), noise_3)

                if alpha4rec != 0:
                    # loss = r_loss
                    loss = nn.L1Loss()
                    Z_opt = noise_weight * noise_1 + noise_3

                    if mode == 'f':
                        _loss = loss(_tmp*zeros, real_curr[3]*zeros)
                        # _loss = calc_local_rec(loss, _tmp, real_seg)
                    elif mode == 'b':
                        _loss = loss(_tmp, real_curr[1])
                    rec_loss = alpha4rec * _loss
                    del _loss
                    # with amp.scale_loss(rec_loss, optimizerG, loss_id=4) as rec_loss:
                    #     rec_loss.backward(retain_graph=True)
                    rec_loss.backward(retain_graph=True)
                    rec_loss = rec_loss.detach()
                else:
                    Z_opt = noise_1
                    rec_loss = torch.Tensor([0])
                if alpha4cos != 0:
                    loss = nn.CosineEmbeddingLoss()
                    Z_opt = noise_weight * noise_1 + noise_3
                    if mode == 'f':
                        _loss = loss(_tmp, real_curr[3], torch.ones_like(real_curr[3]))
                    elif mode == 'b':
                        _loss = loss(_tmp, real_curr[1], torch.ones_like(real_curr[1]))
                    cos_loss = alpha4cos * _loss
                    del _loss
                    # with amp.scale_loss(cos_loss, optimizerG, loss_id=5) as cos_loss:
                    #     cos_loss.backward(retain_graph=True)
                    cos_loss.backward(retain_graph=True)
                    cos_loss = cos_loss.detach()
                else:
                    Z_opt = noise_1
                    cos_loss = torch.Tensor([0])
                if alpha4vgg_ini != 0:
                    loss = p_loss
                    Z_opt = noise_weight * noise_1 + noise_3
                    if mode == 'f':
                        # _loss = alpha4vgg_ini * loss(_tmp, real_curr[3], device)
                        _loss = loss(_tmp, real_curr[3], device)
                    elif mode == 'b':
                        _loss = alpha4vgg_ini * loss(_tmp, real_curr[1], device)
                    perceptual_loss = _loss
                    # perceptual_loss1 = _loss1
                    # perceptual_loss2 = _loss2
                    del _loss
                    # perceptual_loss = factor4vgg[scale_num] * alpha4vgg * p_loss(G(Z_opt.detach(), styles_ref, noise_3), real_curr[3], device)
                    # perceptual_loss = factor4vgg[scale_num] * alpha4vgg * p_loss(G(Z_opt.detach(), noise_3), real_curr[1], device)
                    # with amp.scale_loss(perceptual_loss_f, optimizerG_f, loss_id=6) as perceptual_loss_f:
                    #     perceptual_loss_f.backward(retain_graph=True)
                    # with amp.scale_loss(perceptual_loss, optimizerG, loss_id=5) as perceptual_loss:
                    #     perceptual_loss.backward(retain_graph=True)

                    # perceptual_loss1.backward(retain_graph=True)
                    perceptual_loss.backward(retain_graph=True)
                    # perceptual_loss = perceptual_loss1.detach() + perceptual_loss2.detach()
                    perceptual_loss = perceptual_loss.detach()
                else:
                    Z_opt = noise_1
                    perceptual_loss = torch.Tensor([0])
                optimizerG.step()
                G.zero_grad()
                optimizerG.zero_grad()
            _errG = errG.item()
            _rec_loss = rec_loss.item()
            _cos_loss = cos_loss.item()
            _perceptual_loss = perceptual_loss.item()
            del errG, rec_loss, cos_loss, perceptual_loss
            errG2plot.append([_errG, _rec_loss, _cos_loss, _perceptual_loss])
            _errG = _errG + _rec_loss + _cos_loss + _perceptual_loss
            schedulerG.step(_errG)
            del noise_2
            if i % 200 == 0 or i == (iters-1):
                if mode == 'b':
                    _fake = fake.cpu()
                    _fake = _fake * masks_b[scale_num]
                    _fake = _fake + masks_b[scale_num] - torch.ones_like(masks_b[scale_num])
                    plt.imsave('%s/fake_%s_%s.png' %  (outfile_dir, mode, str(i)), convert_image_np(_fake.detach()), vmin=0, vmax=1)
                elif mode == 'f':
                    plt.imsave('%s/fake_%s_%s.png' %  (outfile_dir, mode, str(i)), convert_image_np(fake.detach()), vmin=0, vmax=1)
            if i % 500 == 0 or i == (iters-1):
                plot_sinloss(errG2plot, errD2plot, scale_num, iters_list, outfile_dir, mode, i)
        epoch_iterator.close()
        torch.save(G.state_dict(), '%s/G_%s.pth' % (outfile_dir, mode))
        torch.save(D.state_dict(), '%s/D_%s.pth' % (outfile_dir, mode))
        G = reset_grads(G, False)
        G.eval()
        D = reset_grads(D, False)
        D.eval()
        Gs.append(G)
        Ds.append(D)
        NoiseWeight.append(noise_weight)
        Zs.append(noise_1)
        # torch.save(Gs, '%s/Gs.pth' % (dir2save))
        torch.save(Zs, '%s/Zs.pth' % (dir2save))
        torch.save(NoiseWeight, '%s/noiseweight_%s.pth' % (dir2save, mode))
        del D, G
    torch.cuda.synchronize()
    end_time = time.time()
    logger.info('-' * 80)
    logger.info(green + '[INFO]: training time cost : %s' % seconds2time(end_time - start_time) + reset)
    logger.info('-' * 80)
    logger.info(green + '[INFO]: randomly generating %s samples...' %(opts.gen_num) + reset)
    logger.info('-' * 80)
    if mode == 'f':
        ran_gen(Gs, Zs, NoiseWeight, reals_fa, opts, dir2gen, padder)
    elif mode == 'b':
        ran_gen(Gs, Zs, NoiseWeight, reals_b, opts, dir2gen, padder)
    logger.info('-' * 80)
    logger.info(green + '[INFO]: calculating eval metrics...' + reset)
    logger.info('-' * 80)
    sifid = calculate_sifid_given_paths(dir2gen+'real.png', dir2gen, batch_size=1, dims=64, suffix='png')
    diversity = calculate_cs(dir2gen, suffix='png')
    logger.info(green + '[INFO]: SIFID : %6f   DIVERSITY : %6f   GQI : %6f ' % (sifid, diversity, diversity/sifid)+ reset)

    # nni.report_final_result(diversity/sifid)

if __name__ == "__main__":
    main()