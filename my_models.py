import torch.nn as nn
import torch
# from dcn_v2 import DCN
class GatedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(GatedConv2d, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.mask_conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.sigmoid = nn.Sigmoid()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
    def gated(self, mask):
        return self.sigmoid(mask)
    def forward(self, x):
        x = self.conv2d(x)
        mask = self.mask_conv2d(x)
        mask = self.gated(mask)
        x = x * mask
        del mask
        return x
class GatedConv2d_Spec(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(GatedConv2d_Spec, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.mask_conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.sigmoid = nn.Sigmoid()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
    def gated(self, mask):
        return self.sigmoid(mask)
    def forward(self, x):
        x = self.conv2d(x)
        mask = self.mask_conv2d(x)
        mask = self.gated(mask)
        # tmp = torch.ones_like(mask)
        # tmp = 6*(mask.round() - tmp)
        # x = x * mask + tmp.detach()
        # del tmp
        return x, mask
class GatedResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(GatedResBlock, self).__init__()
        # self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,stride=stride,padding=int((kernel_size-1)/2))
        # nn.init.kaiming_normal_(self.conv1.weight.data)
        # self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size,stride=stride,padding=int((kernel_size-1)/2))
        # nn.init.kaiming_normal_(self.conv2.weight.data)
        # self.mask_conv = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size,stride=stride,padding=int((kernel_size-1)/2))
        # nn.init.kaiming_normal_(self.mask_conv.weight.data)
        self.gconv1 = GatedConv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.gconv2 = GatedConv2d(out_channels, out_channels, kernel_size, stride, padding)
        self.IN1 = nn.InstanceNorm2d(in_channels)
        self.IN2 = nn.InstanceNorm2d(out_channels)
        # self.BN1 = nn.BatchNorm2d(in_channels)
        # nn.init.normal_(self.BN1.weight.data,1.,0.02)
        # nn.init.zeros_(self.BN1.bias.data)
        # self.BN2 = nn.BatchNorm2d(out_channels)
        # nn.init.normal_(self.BN2.weight.data,1.,0.02)
        # nn.init.zeros_(self.BN2.bias.data)
        self.ELU = nn.ELU(inplace=True)
        # self.leakyrelu = nn.LeakyReLU(0.2, inplace=True)
        self.eca = eca_layer()
        # self.sigmoid = nn.Sigmoid()
        self.model = nn.Sequential(
            # self.BN1,
            self.IN1,
            self.ELU,
            # self.leakyrelu,
            self.gconv1,
            # self.BN2,
            self.IN2,
            self.ELU,
            # self.leakyrelu,
            self.gconv2,
            self.eca
            )
        # self.bypass = nn.Sequential()
        # self.mask_bypass = nn.Sequential(
        #     self.IN1,
        #     self.ELU,
        #     self.mask_conv,
        #     self.sigmoid
        # )
    def forward(self, x):
        # x = self.model(x) * self.mask_bypass(x) + self.bypass(x)
        x1 = self.model(x)
        # x2 = self.bypass(x)
        x1 = x1 + x
        # x = x1 + x2
        del x
        return x1
class GatedResGenerator(nn.Module):
    def __init__(self, channels, out_channels, kernel_size, stride, if_padding, G_num_layer):
        super(GatedResGenerator, self).__init__()
        if if_padding:
            self.padding_size = (kernel_size-1)//2
        else:
            self.padding_size = 0
        self.head = nn.Sequential()
        # self.head.add_module('BN1',nn.BatchNorm2d(channels))
        # nn.init.normal_(self.head[-1].weight.data,1.,0.02)
        # nn.init.zeros_(self.head[-1].bias.data)
        self.head.add_module('IN1', nn.InstanceNorm2d(channels))
        # self.head.add_module('leakyrelu',nn.LeakyReLU(0.2, inplace=True))
        self.head.add_module('elu', nn.ELU(inplace=True))
        self.head.add_module('gconv1', GatedConv2d(channels, out_channels, kernel_size, stride, padding=self.padding_size))
        self.body = nn.Sequential()
        for i in range(G_num_layer):
            self.body.add_module('res_gconv%d'%(i+1), GatedResBlock(out_channels, out_channels, kernel_size, stride, padding=self.padding_size))
        self.tail = nn.Sequential()
        # self.tail.add_module('IN2',nn.InstanceNorm2d(out_channels))
        self.tail.add_module('elu', nn.ELU(inplace=True))
        # self.tail.add_module('leakyrelu',nn.LeakyReLU(0.2, inplace=True))
        # self.tail.add_module('BN2',nn.BatchNorm2d(out_channels))
        # nn.init.normal_(self.tail[-1].weight.data,1.,0.02)
        # nn.init.zeros_(self.tail[-1].bias.data)
        self.tail.add_module('gconv2', GatedConv2d_Spec(out_channels, 3, kernel_size=5, stride=1, padding=2))
        self.act = nn.Tanh()
    def forward(self, x, z, flag=False):
        x = self.head(x)
        x = self.body(x)
        x = self.tail[0](x)
        x, mask = self.tail[-1](x)
        x = self.act(x)
        ind = int((z.shape[2]-x.shape[2])/2)
        z = z[:, :, ind:(z.shape[2]-ind), ind:(z.shape[3]-ind)]
        x = x+z
        del ind, z
        if flag:
            return [x, mask]
        else:
            del mask
            return x

class StyleInjector(nn.Module):
    def __init__(self, if_padding, in_channels=3, out_channels=256, kernel_size=5, padd=0, stride=1):
        super(StyleInjector, self).__init__()
        if if_padding:
            self.padding_size = (kernel_size-1)//2
        else:
            self.padding_size = 0
        self.conv1 = nn.Conv2d(out_channels, out_channels, kernel_size=5, stride=stride, padding=self.padding_size)
        nn.init.kaiming_normal_(self.conv1.weight.data)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=5, stride=stride, padding=self.padding_size)
        nn.init.kaiming_normal_(self.conv2.weight.data)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=5, stride=stride, padding=self.padding_size)
        nn.init.kaiming_normal_(self.conv3.weight.data)
        self.conv4 = nn.Conv2d(out_channels, out_channels, kernel_size=5, stride=stride, padding=self.padding_size)
        nn.init.kaiming_normal_(self.conv4.weight.data)
        # self.dcn = DCN(in_channels, out_channels, kernel_size=5, padding=2, stride=stride)
        # nn.init.kaiming_normal_(self.dcn.weight.data)
        # self.dcn1 = DCN(in_channels, out_channels, kernel_size=5, padding=2, stride=stride)
        # nn.init.kaiming_normal_(self.dcn1.weight.data)
        # self.dcn2 = DCN(out_channels, out_channels, kernel_size=5, padding=2, stride=stride)
        # nn.init.kaiming_normal_(self.dcn2.weight.data)
        # self.dcn3 = DCN(in_channels, out_channels, kernel_size=5, padding=2, stride=stride)
        # nn.init.kaiming_normal_(self.dcn1.weight.data)
        # self.dcn4 = DCN(out_channels, out_channels, kernel_size=5, padding=2, stride=stride)
        # nn.init.kaiming_normal_(self.dcn2.weight.data)
        self.conv = nn.Conv2d(in_channels, out_channels, 5, 1, self.padding_size, bias=False)
        nn.init.kaiming_normal_(self.conv.weight.data)
        self.BN1 = nn.BatchNorm2d(out_channels)
        nn.init.normal_(self.BN1.weight.data, 1., 0.02)
        nn.init.zeros_(self.BN1.bias.data)
        self.BN2 = nn.BatchNorm2d(out_channels)
        nn.init.normal_(self.BN2.weight.data, 1., 0.02)
        nn.init.zeros_(self.BN2.bias.data)
        self.BN3 = nn.BatchNorm2d(out_channels)
        nn.init.normal_(self.BN3.weight.data, 1., 0.02)
        nn.init.zeros_(self.BN3.bias.data)
        self.BN4 = nn.BatchNorm2d(out_channels)
        nn.init.normal_(self.BN4.weight.data, 1., 0.02)
        nn.init.zeros_(self.BN4.bias.data)
        self.BN5 = nn.BatchNorm2d(out_channels)
        nn.init.normal_(self.BN5.weight.data, 1., 0.02)
        nn.init.zeros_(self.BN5.bias.data)
        # self.ELU = nn.ELU(inplace=True)
        self.leakyrelu = nn.LeakyReLU(0.2, inplace=True)
        # self.esa = esa_layer()
        self.eca1 = eca_layer()
        self.eca2 = eca_layer()
        self.weight_bypass = nn.Sequential(
            self.BN1,
            # self.ELU,
            self.leakyrelu,
            self.conv1,
            self.BN2,
            self.leakyrelu,
            # self.ELU,
            self.conv2,
            self.eca1,
            nn.Tanh(),
            # self.ELU
            # self.esa
        )
        self.bias_bypass = nn.Sequential(
            self.BN3,
            # self.ELU,
            self.leakyrelu,
            self.conv3,
            self.BN4,
            self.leakyrelu,
            self.conv4,
            self.eca2,
            # self.ELU,
            nn.Tanh(),
            # self.esa
        )
        # self.weight_bypass = nn.Sequential(
        #     self.BN1,
        #     self.ELU,
        #     self.dcn1,
        #     # self.esa,
        # )
        # self.bias_bypass = nn.Sequential(
        #     self.BN1,
        #     self.ELU,
        #     self.dcn2,
        #     # self.esa,
        # )
        self.bypass = nn.Sequential(
            self.conv,
            # self.BN5,
            # self.leakyrelu
        )
        # self.bypass = nn.Sequential()
    def forward(self, x):
        x = self.bypass(x)
        w = self.weight_bypass(x)
        w = w + x[:, :, 4:-4, 4:-4] if self.padding_size == 0 else w + x
        b = self.bias_bypass(x)
        b = b + x[:, :, 4:-4, 4:-4] if self.padding_size == 0 else b + x
        x = [w, b]
        del w, b
        # del x
        return x
class eca_layer(nn.Module):
    def __init__(self, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y).expand_as(x)
        x = x*y
        del y
        return x

class styleconv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, if_padding, stride, factor):
        super(styleconv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding_size = (self.kernel_size-1)//2 if if_padding else 0
        self.stride = stride
        self.conv = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=self.kernel_size, padding=self.padding_size, stride=self.stride)
        nn.init.kaiming_normal_(self.conv.weight.data)
        self.IN = nn.InstanceNorm2d(in_channels)
        self.leakyrelu = nn.LeakyReLU(0.2, inplace=True)
        self.eca = eca_layer()
        self.factor = factor
    def forward(self, x, y):
        x = self.IN(x)
        x = y[0] * x + y[1]
        x = self.factor * x
        x = self.leakyrelu(x)
        x = self.conv(x)
        # x = self.eca(x)
        return x
class styleResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, if_padding, stride, factor):
        super(styleResBlock, self).__init__()
        self.styleconv1 = styleconv(in_channels, out_channels, kernel_size, if_padding, stride, factor)
        self.styleconv2 = styleconv(in_channels, out_channels, kernel_size, if_padding, stride, factor)
        self.dropout = nn.Dropout(p=0.1)
    def forward(self, x, y1, y2):
        x_ = self.styleconv1(x, y1)
        x_ = self.styleconv2(x, y2)
        x_ = self.dropout(x_ + x)
        # x_ = x_ + x
        return x_

# class newResAttnBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, padding_size, stride, dilation=1, groups=1):
#         super(newResAttnBlock, self).__init__()
#         self.BN1 = nn.BatchNorm2d(in_channels)
#         nn.init.normal_(self.BN1.weight.data, 1., 0.02)
#         nn.init.zeros_(self.BN1.bias.data)
#         self.BN2 = nn.BatchNorm2d(in_channels)
#         nn.init.normal_(self.BN2.weight.data, 1., 0.02)
#         nn.init.zeros_(self.BN2.bias.data)
#         self.IN = nn.InstanceNorm2d(in_channels)
#         self.leakyrelu = nn.LeakyReLU(0.2, inplace=True)
#         self.dcn = DCN(in_channels, in_channels, kernel_size=kernel_size, padding=padding_size, stride=stride)
#         self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=padding_size, stride=stride)
#         self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=padding_size, stride=stride)
#         nn.init.kaiming_normal_(self.dcn.weight.data)
#         nn.init.kaiming_normal_(self.conv.weight.data)
#         nn.init.kaiming_normal_(self.conv2.weight.data)
#         self.eca = eca_layer()
#         self.dcn_bypass = nn.Sequential(
#             self.IN,
#             # self.BN1,
#             # self.ELU,
#             self.leakyrelu,
#             self.conv,
#             self.IN,
#             # self.BN2,
#             # self.ELU,
#             self.leakyrelu,
#             # self.dcn,
#             self.conv2,
#             self.eca
#         )
#         # self.bypass = nn.Sequential()
#     def forward(self, x):
#         x1 = self.dcn_bypass(x)
#         x1 = x1 + x[:, :, 2:-2, 2:-2]
#         # x2 = self.bypass(x)
#         # x = x1 + x2
#         del x
#         return x1
class ResGenerator(nn.Module):
    def __init__(self, channels, out_channels, kernel_size, stride, if_padding, G_num_layer, factor=1):
        super(ResGenerator, self).__init__()
        if if_padding:
            self.padding_size = (kernel_size-1)//2
        else:
            self.padding_size = 0
        self.is_cuda = torch.cuda.is_available()
        self.factor = factor
        self.G_num_layer = G_num_layer
        # self.swish = Swish()
        # self.ELU = nn.ELU(inplace=True)
        self.IN = nn.InstanceNorm2d(out_channels)
        self.head = nn.Sequential()
        # self.head.add_module('BN1',nn.BatchNorm2d(channels))
        # nn.init.normal_(self.head[-1].weight.data,1.,0.02)
        # nn.init.zeros_(self.head[-1].bias.data)
        self.head.add_module('IN1', nn.InstanceNorm2d(channels))
        self.head.add_module('leakyrelu', nn.LeakyReLU(0.2, inplace=True))
        # self.head.add_module('swish',self.swish)
        # self.head.add_module('elu',nn.ELU(inplace=True))
        # self.head.add_module('conv1',nn.Conv2d(channels, out_channels, kernel_size=3,stride=stride,padding=int((kernel_size-1)/2)))
        self.head.add_module('conv1', nn.Conv2d(channels, out_channels, kernel_size, stride, self.padding_size))
        nn.init.kaiming_normal_(self.head[-1].weight.data)
        self.body = nn.Sequential()
        self.injector = StyleInjector(if_padding)
        self.body.add_module('styleconv1', styleconv(out_channels, out_channels, kernel_size, if_padding, stride, factor))
        for i in range(self.G_num_layer):
            # self.body.add_module('res%d'%(i+1),ResBlockGenerator(out_channels, out_channels, kernel_size, padding_size, stride))
            # self.body.add_module('dconv%d'%(i+1),ResAttnBlock(out_channels, out_channels, kernel_size, padding_size, stride))
            # self.body.add_module('new_res%d'%(i+1), newResAttnBlock(out_channels, out_channels, kernel_size, self.padding_size, stride))
            # self.body.add_module('styleconv%d'%(i+2), styleconv(out_channels, out_channels, kernel_size, if_padding, stride, factor))
            self.body.add_module('styleresblock%d'%(i+1), styleResBlock(out_channels, out_channels, kernel_size, if_padding, stride, factor))
            # self.body.add_module('res%d'%(2*i+1),ResBlockGenerator(out_channels, out_channels, kernel_size, padding_size, stride))
            # self.body.add_module('dropout%d'%(i+1),self.dropout)
            # self.injector.add_module('sty_inj%d'%(i+1),StyleInjector())
        # self.res_block1 = ResBlockGenerator(out_channels, out_channels, kernel_size, padding_size, stride)
        # self.res_block2 = ResBlockGenerator(out_channels, out_channels, kernel_size, padding_size, stride)
        # self.res_block3 = ResBlockGenerator(out_channels, out_channels, kernel_size, padding_size, stride)
        # self.res_attn_block1 = ResAttnBlock(out_channels, out_channels, kernel_size, padding_size, stride)
        # self.res_attn_block2 = ResAttnBlock(out_channels, out_channels, kernel_size, padding_size, stride)
        # self.res_attn_block3 = ResAttnBlock(out_channels, out_channels, kernel_size, padding_size, stride)
        self.body.add_module('styleconv2', styleconv(out_channels, channels, kernel_size=5, if_padding=if_padding, stride=stride, factor=self.factor))
        self.tail = nn.Sequential()
        # self.tail.add_module('BN2',nn.BatchNorm2d(out_channels))
        # nn.init.normal_(self.tail[-1].weight.data,1.,0.02)
        # nn.init.zeros_(self.tail[-1].bias.data)
        # self.tail.add_module('IN2',nn.InstanceNorm2d(out_channels))
        # self.tail.add_module('leakyrelu', nn.LeakyReLU(0.2, inplace=True))
        # self.tail.add_module('swish',self.swish)
        # self.tail.add_module('elu',nn.ELU(inplace=True))
        # self.tail.add_module('conv2',nn.Conv2d(out_channels, channels, kernel_size=3,stride=stride,padding=int((kernel_size-1)/2)))
        # if if_padding:
            # self.tail.add_module('conv2', nn.Conv2d(out_channels, channels, kernel_size=5, stride=1, padding=2))
        # else:
            # self.tail.add_module('conv2', nn.Conv2d(out_channels, channels, kernel_size=5, stride=1, padding=0))
        # nn.init.kaiming_normal_(self.tail[-1].weight.data)
        self.tail.add_module('act', nn.Tanh())
    def forward(self, x, y, z):
        x = self.head(x)
        x = self.body[0](x, self.injector(y[0]))
        for i in range(self.G_num_layer):
            x = self.body[i+1](x, self.injector(y[2*i+1]), self.injector(y[2*i+2]))
            # x1 = tmp[0]*x
            # x2 = tmp[1]
            # del tmp
            # x = self.factor  * (x1 + x2)
            # del x1, x2
            # x = x + self.factor*tmp
            # del tmp
            # x = self.body[2*i](x)
            # x = self.body[2*i+1](x)
            # x = self.body[i](x)
        # x = self.body(x)
        x = self.body[-1](x, self.injector(y[-1]))
        x = self.tail(x)
        ind = int((z.shape[2]-x.shape[2])/2)
        z = z[:, :, ind:(z.shape[2]-ind), ind:(z.shape[3]-ind)]
        del ind
        return x+z


class FcDiscriminator(nn.Module):
    def __init__(self, channels, out_channels, kernel_size, stride, padding_size, D_num_layer, if_BN=True):
        super(FcDiscriminator, self).__init__()
        self.is_cuda = torch.cuda.is_available()
        self.if_BN = if_BN
        self.head = DConvBlock(channels, out_channels, kernel_size=5, padd=2, stride=1, if_BN=self.if_BN)
        self.body = nn.Sequential()
        for i in range(D_num_layer - 2):
            block = DConvBlock(out_channels, out_channels, kernel_size, padding_size, stride, if_BN=self.if_BN)
            self.body.add_module('block%d'%(i+1), block)
        self.tail = nn.Conv2d(out_channels, 1, kernel_size, stride, padding_size)
    def forward(self, x):
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        return x
class DConvBlock(nn.Sequential):
    def __init__(self, in_channel, out_channel, kernel_size, padd, stride, if_BN=True):
        super(DConvBlock, self).__init__()
        self.if_BN = if_BN
        self.add_module('conv', nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padd))
        nn.init.kaiming_normal_(self.conv.weight.data)
        if self.if_BN:
            self.add_module('norm', nn.BatchNorm2d(out_channel))
            nn.init.normal_(self.norm.weight.data, 1., 0.02)
            nn.init.zeros_(self.norm.bias.data)
        self.add_module('LeakyRelu', nn.LeakyReLU(0.2, inplace=True))
        # self.add_module('elu',nn.ELU(inplace=True))