import math
import xml.etree.ElementTree as ET
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import colorama
import matplotlib
import matplotlib.pyplot as plt
from skimage import color
from torchvision import transforms
from progress_bar import create_progressbar
from PIL import Image
from skimage.segmentation import felzenszwalb, slic
matplotlib.use('Agg')
# from skimage import io as img
# from skimage import transform, img_as_uint

def get_seg(img, _size, n_segments=50, compactness=10, sigma=1, start_label=1):
    edge_labs = []
    img = img.to("cpu") if img.is_cuda else img
    img = F.interpolate(img, _size, mode='bicubic', align_corners=False)
    img = img[0, :, :, :].transpose((1, 2, 0))
    img = img.numpy()
    segs = slic(img, n_segments=n_segments, compactness=compactness, sigma=sigma, start_label=start_label, max_iter=50)
    _w, _h = _size[1], _size[0]
    return segs
def calc_local_rec():
    return None

def modify_scales(anot_path, scale_base):
    corners = []
    tree = ET.parse(anot_path)
    root = tree.getroot()
    for obj in root.findall('object'):
        for  bndbox in obj.findall('bndbox'):
            xmin = int(bndbox[0].text)
            ymin = int(bndbox[1].text)
            xmax = int(bndbox[2].text)
            ymax = int(bndbox[3].text)
            corners.append([xmin, ymin, xmax, ymax])
    w = corners[0][2] - corners[0][0]
    h = corners[0][3] - corners[0][1]
    scales = 0
    while True:
        w_ = int(w * math.pow(scale_base, scales))
        h_ = int(h * math.pow(scale_base, scales))
        if w_ <= 20 or h_ <= 20:
            break
        scales += 1
    return scales+1
def ran_gen(Gs, Zs, NoiseWeight, reals, opts, dir2gen, padder):
    images_curr = []
    n = 0
    scales = len(Gs)
    for G, Z, noiseweight, real in zip(Gs, Zs, NoiseWeight, reals):
        assert not G.training
        processbar = create_progressbar(
            iterable=range(opts.gen_num),
            desc="Generating scale [{}/{}]".format(n, scales-1),
            offset=0, leave=True, logging_on_update=False, logging_on_close=True, postfix=True
        )
        images_prev = images_curr
        images_curr = []
        h, w = real.shape[2], real.shape[3]
        h, w = int(h*opts.scale_h), int(w*opts.scale_w)
        real = F.interpolate(real, [h, w], mode='bicubic', align_corners=False)
        plt.imsave(dir2gen+'real.png', convert_image_np(real), vmin=0, vmax=1)
        real = real.squeeze(0)
        for i in processbar:
            processbar.set_description('Generating scale [{}/{}], Num [{}/{}]'.format(n+1, scales, i+1, opts.gen_num))
            if n == 0:
                z_curr = padder(Generate_noise([1, h, w], torch.device("cuda:0"), False, True))
            else:
                z_curr = padder(Generate_noise([3, h, w], torch.device("cuda:0"), False, False))
            if images_prev == []:
                I_prev = torch.zeros_like(z_curr).to(torch.device("cuda:0"))
            else:
                I_prev = images_prev[i]
                I_prev = F.interpolate(I_prev, [h, w], mode='bicubic', align_corners=False)
                I_prev = padder(I_prev)
            # if n<start_scale:
            #     z_curr = Z_f
            if opts.mode == 'f':
                data_aug = []
                # plt.imsave(outputdir+'/a.png', convert_image_np(_augument.detach()), vmin=0, vmax=1)
                for j in range(opts.G_num_layer*2+2):
                    _padder = make_padder(2+2*3+(opts.G_num_layer-1-j)*1) if padder.padding[0] != 0 else make_padder(0)
                    _augment = data_augmenter(real, torch.device("cuda:0"))
                    _augment = _augment.detach()
                    _augment_ = _padder(_augment)
                    data_aug.append(_augment_)
            z_in = noiseweight * z_curr + I_prev
            if opts.mode == 'f':
                I_curr = G(z_in.detach(), data_aug, I_prev)
            elif opts.mode == 'b':
                I_curr = G(z_in.detach(), I_prev)
            I_curr = I_curr.detach()
            images_curr.append(I_curr)
            plt.imsave(dir2gen+'fake_%s.png' %  (i), convert_image_np(I_curr), vmin=0, vmax=1)
        del G
        n += 1

def seconds2time(seconds):
    t_m, t_s = divmod(seconds, 60)
    t_h, t_m = divmod(t_m, 60)
    r_t = str(int(t_h)).zfill(2) + ":" + str(int(t_m)).zfill(2) + ":" + str(int(t_s)).zfill(2)
    return r_t

def get_logger():
    reset = colorama.Style.RESET_ALL
    dim = colorama.Style.DIM
    white = colorama.Fore.WHITE
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter(
            fmt="{}{}[%(asctime)s]{} %(message)s".format(dim, white, reset),
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        console.setFormatter(formatter)
        logger.addHandler(console)
    return logger

def data_augmenter(x, device):
    h, w = x.shape[1], x.shape[2]
    transformer = transforms.Compose([
        transforms.ToPILImage(),
        # transforms.RandomGrayscale(p=1),
        # transforms.Resize((2*h,2*w)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomChoice([
            transforms.RandomRotation(45),
            transforms.RandomPerspective(0.2)
        ]),
        transforms.RandomChoice([
            transforms.RandomAffine(degrees=0, translate=(0.2, 0.2)),
            transforms.RandomResizedCrop((h, w), (0.7, 0.9))
        ]),
        transforms.ToTensor(),
        transforms.RandomErasing(),
    ])
    x = transformer(x)
    x = x[None, :, :, :][:, 0:3, :, :]
    x = (x*2-1).clamp(-1, 1).to(device).type(torch.cuda.FloatTensor)
    return x

def plot_sinloss(err_G, err_D, scale_num, iters_list, output_dir, flag, iter_curr):
    colors = [
        '#1f77b4',
        '#ff7f0e',
        '#2ca02c',
        '#d62728',
        '#9467bd',
        '#8c564b',
        '#e377c2',
        '#7f7f7f',
        '#bcbd22',
        '#17becf',
        '#1a55FF'
    ]

    x = np.arange(0, iter_curr, 1)
    _ = 0
    if scale_num != 0:
        for i in iters_list[0:scale_num]:
            _ = _ + i
    err_G, err_D = np.array(err_G[_:_+iter_curr]), np.array(err_D[_:_+iter_curr])
    fig, axs = plt.subplots(2, 1)

    rec, cos, vgg, g_fake = [], [], [], []
    for i in err_G:
        g_fake.append(i[0])
        rec.append(i[1])
        cos.append(i[2])
        vgg.append(i[3])
    d_real, d_fake, gp = [], [], []
    for i in err_D:
        d_real.append(i[0])
        d_fake.append(i[1])
        gp.append(i[2])

    axs[0].plot(x, g_fake, color=colors[0], label='g_fake')
    axs[0].plot(x, rec, color=colors[1], label='rec')
    axs[0].plot(x, cos, color=colors[2], label='cos')
    axs[0].plot(x, vgg, color=colors[3], label='perceptual')
    axs[0].legend()
    axs[0].set_xlabel('iters')
    axs[0].set_ylabel('err_G')
    axs[0].grid(True)

    axs[1].plot(x, d_real, color=colors[4], label='d_real')
    axs[1].plot(x, d_fake, color=colors[5], label='d_fake')
    axs[1].plot(x, gp, color=colors[6], label='gp')
    axs[1].legend()
    axs[1].set_xlabel('iters')
    axs[1].set_ylabel('err_D')
    axs[1].grid(True)

    plt.savefig("%s/%s%s_loss.png" % (output_dir, flag, str(scale_num)))

def make_padder(padsize):
    # padsize = int(((kernel_size - 1) * num_layer) / 2) + 1 + 2 #使用valid padding 首尾分别一个3和5的核
    padder = nn.ZeroPad2d(padsize)
    # padder = nn.ReflectionPad2d(padsize)
    return padder

def get_reals(mode, img_path, anot_path, scales, scale_base, reals, channels, masks):
    # import pdb;pdb.set_trace()
    # x = img.imread(img_path)
    x = Image.open(img_path)
    # w, h = x.shape[1], x.shape[0]
    w, h = x.size[0], x.size[1]

    corners = []
    tree = ET.parse(anot_path)
    root = tree.getroot()
    for obj in root.findall('object'):
        for  bndbox in obj.findall('bndbox'):
            xmin = int(bndbox[0].text)
            ymin = int(bndbox[1].text)
            xmax = int(bndbox[2].text)
            ymax = int(bndbox[3].text)
            corners.append([xmin, ymin, xmax, ymax])
    mask_f = np.zeros_like(np.array(x)[:, :, 0])
    for corner in corners:
        mask_f[corner[1]:corner[3]+1, corner[0]:corner[2]+1] += 1
    mask_b = 1 - mask_f
    # mask_b, mask_f = img_as_uint(mask_b), img_as_uint(mask_f)
    mask_b, mask_f = Image.fromarray(mask_b.astype('bool_')).convert('1'), Image.fromarray(mask_f.astype('bool_')).convert('1')
    for i in range(scales):
        # w_, h_ = int(w * math.pow(scale_base, ( ((scales-1)/math.log(scales))*math.log(scales-i) ))), int(h*math.pow(scale_base, ( ((scales-1)/math.log(scales))*math.log(scales-i) )))
        # w_min, h_min = int(w*math.pow(scale_base,scales-1)), int(h*math.pow(scale_base,scales-1))
        # w_, h_ = w_min + int((w-w_min)/(scales-1)*i), h_min + int((h-h_min)/(scales-1)*i)
        w_, h_ = int(w*math.pow(scale_base, i)), int(h*math.pow(scale_base, i))
        resizer = transforms.Resize((h_, w_), interpolation=5)
        # x_, mask_b_, mask_f_ = transform.resize(x, (h_, w_), anti_aliasing=True), transform.resize(mask_b, (h_, w_)), transform.resize(mask_f, (h_, w_))
        x_, mask_b_, mask_f_ = resizer(x), resizer(mask_b), resizer(mask_f)
        x_, mask_b_, mask_f_ = np.array(x_, dtype=np.float64), np.array(mask_b_, dtype=np.float64), np.array(mask_f_, dtype=np.float64)
        if channels == 3:
            # x_, mask_b_, mask_f_ = x_[:, :, :, None], mask_b_[:, :, :, None], mask_f_[:, :, :, None]
            # x_, mask_b_, mask_f_ = x_.transpose((3, 2, 0, 1)), mask_b_.transpose((3, 2, 0, 1)), mask_f_.transpose((3, 2, 0, 1))
            mask_b_, mask_f_ = np.expand_dims(mask_b_, -1).repeat(3, axis=-1), np.expand_dims(mask_f_, -1).repeat(3, axis=-1)
            x_, mask_b_, mask_f_ = x_[:, :, :, None], mask_b_[:, :, :, None], mask_f_[:, :, :, None]
            x_, mask_b_, mask_f_ = x_.transpose((3, 2, 0, 1)), mask_b_.transpose((3, 2, 0, 1)), mask_f_.transpose((3, 2, 0, 1))
        else:
            #暂不支持
            print("?")
            x_ = color.rgb2gray(x_)
            x_ = x_[:, :, None, None]
            x_ = x_.transpose(3, 2, 0, 1)
        x_, mask_b_, mask_f_ = x_[:, 0:3, :, :], mask_b_[:, 0:3, :, :], mask_f_[:, 0:3, :, :]
        # mask_b_, mask_f_ = np.ceil(mask_b_), np.ceil(mask_f_)

        x_, mask_b_, mask_f_ = torch.from_numpy(x_), torch.from_numpy(mask_b_), torch.from_numpy(mask_f_)
        x_, mask_b_, mask_f_ = x_.type(torch.FloatTensor), mask_b_.type(torch.FloatTensor), mask_f_.type(torch.FloatTensor)
        x_ = x_ / 255
        x_ = (x_ - 0.5)*2
        x_ = x_.clamp(-1, 1)
        fs = mask_f_*x_
        b = mask_b_*x_
        b, fs = b+(mask_b_ - 1), fs+(mask_f_ - 1)

        _ = np.where(mask_f_ != 0)
        fa = fs[:, :, np.min(_[2]):np.max(_[2]), np.min(_[3]):np.max(_[3])]

        reals.append([x_, b, fs, fa])
        masks.append([mask_b_, mask_f_])

    return reals, masks

def weights_init(m):
    classname = m.__class__.__name__
#     print(classname)
#     print(classname.find('Conv2d'))
    # print(classname.find('Norm'))
    if classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
        # m.weight.data.kaiming_normal_
    elif classname.find('Norm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    else:
        pass

def Generate_noise(size, device, if_0, if_c_same, num_samp=1, type='gaussian', scale=1, channels=3):
    if type == 'gaussian':
        noise = torch.randn(num_samp, size[0], round(size[1]/scale), round(size[2]/scale), device=device)
        m = nn.Upsample(size=[round(size[1]), round(size[2])], mode='bicubic', align_corners=False)
        noise = m(noise)
    if type == 'gaussian_mixture':
        noise1 = torch.randn(num_samp, size[0], size[1], size[2], device=device)+5
        noise2 = torch.randn(num_samp, size[0], size[1], size[2], device=device)
        noise = noise1+noise2
    if type == 'uniform':
        noise = torch.randn(num_samp, size[0], size[1], size[2], device=device)
    if if_0:
        noise = torch.full(noise.shape, 0, device=device)
    if if_c_same:
        noise = noise.expand(size[0], channels, size[1], size[2])
    else:
        noise = noise.expand(1, size[0], size[1], size[2])
    return noise

def generate_noise(size, device, num_samp=1, type='gaussian', scale=1):
        #仅在draw_concat中被使用
    if type == 'gaussian':
        noise = torch.randn(num_samp, size[0], round(size[1]/scale), round(size[2]/scale), device=device)
        m = nn.Upsample(size=[round(size[1]), round(size[2])], mode='bicubic', align_corners=False)
        noise = m(noise)
    if type == 'gaussian_mixture':
        noise1 = torch.randn(num_samp, size[0], size[1], size[2], device=device)+5
        noise2 = torch.randn(num_samp, size[0], size[1], size[2], device=device)
        noise = noise1+noise2
    if type == 'uniform':
        noise = torch.randn(num_samp, size[0], size[1], size[2], device=device)
    return noise

def convert_image_np(inp):
    if inp.shape[1] == 3:
        inp = ((inp + 1)/2).clamp(0, 1)
        inp = inp[-1, :, :, :]
        inp = inp.to(torch.device('cpu'))
        inp = inp.numpy().transpose((1, 2, 0))
    else:
        # print('???')
        inp = ((inp + 1)/2).clamp(0, 1)
        inp = inp[-1, -1, :, :].cpu()
        inp = inp.numpy().transpose((0, 1))
        # mean = np.array([x/255.0 for x in [125.3,123.0,113.9]])
        # std = np.array([x/255.0 for x in [63.0,62.1,66.7]])
    inp = np.clip(inp, 0, 1)
    return inp

# @profile
def draw_concat(Gs, Zs, reals, NoiseWeight, in_s, mode, kernel_size, channels, device, padder, G_num_layer, flag):
    G_z = in_s
    if len(Gs) > 0:
        if mode == 'rand':
            count = 0
            pad_noise = padder.padding[0]
            for G, Z_opt, real_curr, real_next, noise_weight in zip(Gs, Zs, reals, reals[1:], NoiseWeight):
                if count == 0:
                    z = generate_noise([1, Z_opt.shape[2] - 2 * pad_noise, Z_opt.shape[3] - 2 * pad_noise], device=device)
                    z = z.expand(1, 3, z.shape[2], z.shape[3])
                else:
                    z = generate_noise([channels, Z_opt.shape[2] - 2 * pad_noise, Z_opt.shape[3] - 2 * pad_noise], device=device)
                z = padder(z)
                G_z = G_z[:, :, 0:real_curr.shape[2], 0:real_curr.shape[3]]
                G_z = padder(G_z)
                z_in = noise_weight*z+G_z
                del z
                if flag == 'f':
                    styles_ref = []
                    for _ in range(G_num_layer*2+2):
                        if pad_noise == 0:
                            _padder = nn.ZeroPad2d(0)
                        else:
                            _padder = nn.ZeroPad2d(2+2*3+(G_num_layer-1-_)*1)
                        # _padder = nn.ReflectionPad2d(2+2*3+(G_num_layer-_)*2)
                        _augment = data_augmenter(real_curr.squeeze(0).cpu(), device=device)
                        _augment_ = _padder(_augment)
                        styles_ref.append(_augment_.detach())
                    del _augment, _augment_
                    G = G.to(device)
                    G_z = G(z_in.detach(), styles_ref, G_z)
                elif flag == 'b':
                    G = G.to(device)
                    G_z = G(z_in.detach(), G_z)
                del G, real_curr
                G_z = F.interpolate(G_z, [real_next.shape[2], real_next.shape[3]], mode='bicubic', align_corners=False)
                count += 1
        if mode == 'rec':
            pad_noise = padder.padding[0]
            count = 0
            for G, Z_opt, real_curr, real_next, noise_weight in zip(Gs, Zs, reals, reals[1:], NoiseWeight):
                G_z = G_z[:, :, 0:real_curr.shape[2], 0:real_curr.shape[3]]
                G_z = padder(G_z)
                z_in = noise_weight*Z_opt+G_z
                if flag == 'f':
                    styles_ref = []
                    for _ in range(G_num_layer*2+2):
                        if pad_noise == 0:
                            _padder = nn.ZeroPad2d(0)
                        else:
                            _padder = nn.ZeroPad2d(2+2*3+(G_num_layer-1-_)*1)
                        # _padder = nn.ReflectionPad2d(2+2*3+(G_num_layer-_)*2)
                        _augment = data_augmenter(real_curr.squeeze(0).cpu(), device=device)
                        _augment_ = _padder(_augment)
                        styles_ref.append(_augment_.detach())
                    del _augment, _augment_
                    G = G.to(device)
                    G_z = G(z_in.detach(), styles_ref, G_z)
                elif flag == 'b':
                    G = G.to(device)
                    G_z = G(z_in.detach(), G_z)
                del G, real_curr
                G_z = F.interpolate(G_z, [real_next.shape[2], real_next.shape[3]], mode='bicubic', align_corners=False)
                count += 1
    return G_z

def calc_gradient_penalty(netD, real_data, fake_data, LAMBDA, device):
    alpha = torch.rand(1, 1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.to(device)#cuda() #gpu) #if use_cuda else alpha
    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    interpolates = interpolates.to(device)#.cuda()
    interpolates = torch.autograd.Variable(interpolates, requires_grad=True)
    disc_interpolates = netD(interpolates)
    gradients = torch.autograd.grad(
        outputs=disc_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    #LAMBDA = 1
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty

def reset_grads(model, require_grad):
    for p in model.parameters():
        p.requires_grad_(require_grad)
    return model