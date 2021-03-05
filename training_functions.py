import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from my_models import ResGenerator, FcDiscriminator, GatedResGenerator
# from adabelief_pytorch import AdaBelief

def organise_models(mode, device, weights2load, lr_g, lr_d, channels, kernel_size, stride, if_padding, G_num_layer, D_num_layer, out_channels, factor=None):

    # out_channels_base = 64
    beta4adam = 0.5
    gamma = 0.1
    # default : stride=1
    padding_size = (kernel_size-1)//2
    if mode == 'f':
        G = ResGenerator(channels, out_channels, kernel_size, stride, if_padding, G_num_layer, factor).to(device)
        D = FcDiscriminator(channels, out_channels, kernel_size, stride, padding_size, D_num_layer).to(device)
    elif mode == 'b':
        G = GatedResGenerator(channels, out_channels, kernel_size, stride, if_padding, G_num_layer).to(device)
        D = FcDiscriminator(channels, out_channels, kernel_size, stride, padding_size, D_num_layer, if_BN=False).to(device)

    optimizerD = optim.Adam(D.parameters(), lr=lr_d, betas=(beta4adam, 0.999))
    optimizerG = optim.Adam(G.parameters(), lr=lr_g, betas=(beta4adam, 0.999))
    # optimizerD = AdaBelief(D.parameters(), lr=lr_d, eps=1e-12, betas=(beta4adam, 0.999))
    # optimizerG = AdaBelief(G.parameters(), lr=lr_g, eps=1e-12, betas=(beta4adam, 0.999))

    schedulerD = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizerD, milestones=[1600], gamma=gamma)
    schedulerG = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizerG, milestones=[1600], gamma=gamma)
    # schedulerD = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizerD,mode='min',factor=0.5,patience=3000)
    # schedulerG = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizerG,mode='min',factor=0.5,patience=3000)

    return D, G, optimizerD, optimizerG, schedulerD, schedulerG

class recon_loss(torch.nn.Module):
    def __init__(self, flag, gamma=2):
        super(recon_loss, self).__init__()
        self.flag = flag
        self.gamma = gamma
        self.loss = nn.MSELoss()
    def forward(self, x, y):
        if self.flag:
            se = (x-y)**2
            min_, max_ = torch.min(se, 1, True).values.expand_as(se), torch.max(se, 1, True).values.expand_as(se)
            se = (se-min_)/(max_-min_)
            _ = self.loss((x * F.sigmoid(se)), y)
        else:
            _ = self.loss(x, y)
        return _

class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, device, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval().to(device))
        blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval().to(device))
        blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval().to(device))
        blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval().to(device))
        for bl in blocks:
            for p in bl:
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        # self.mean = torch.nn.Parameter(torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        # self.std = torch.nn.Parameter(torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))
        # self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        # self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        # self.mean, self.std = self.mean.to(device), self.std.to(device)
        self.resize = resize

    def forward(self, x, y, device):
        if x.shape[1] != 3:
            x = x.repeat(1, 3, 1, 1)
            y = y.repeat(1, 3, 1, 1)
        # x = (x-self.mean) / self.std
        # y = (y-self.mean) / self.std
        if self.resize:
            x = self.transform(x, mode='bicubic', size=(224, 224), align_corners=False)
            y = self.transform(y, mode='bicubic', size=(224, 224), align_corners=False)
        # loss1 = 0.0
        loss2 = 0.0
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            # if i == 0:
            #     loss1 += torch.nn.functional.l1_loss(x, y)
            if i >= 3: # 0 1 2 3
                loss2 += torch.nn.functional.l1_loss(x, y)
        return loss2
def make_out_channels(out_channels_base, num_layer, mode):
    out_channels = []
    if mode == 'g':
        out_channels.append(out_channels_base)
        for i in range(num_layer):
            out_channels.append(out_channels_base*2**(i+1) if out_channels_base*2**(i+1) <= 512 else 512)
        _ = out_channels[::-1]
        out_channels += _[1:]
    elif mode == 'd':
        out_channels = 256
    return out_channels