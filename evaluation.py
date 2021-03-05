import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import os
import pathlib
import numpy as np
import torch
from scipy import linalg
from matplotlib.pyplot import imread
from torch.nn.functional import adaptive_avg_pool2d
import torchvision
import scipy
import pickle

def get_activations(files, model, batch_size=1, dims=64,
                    verbose=False):
    model.eval()

    if len(files) % batch_size != 0:
        print(('Warning: number of images is not a multiple of the '
               'batch size. Some samples are going to be ignored.'))
    if batch_size > len(files):
        print(('Warning: batch size is bigger than the data size. '
               'Setting batch size to data size'))
        batch_size = len(files)
    n_batches = len(files) // batch_size
    n_used_imgs = n_batches * batch_size
    pred_arr = np.empty((n_used_imgs, dims))
    for i in range(n_batches):
        if verbose:
            print('\rPropagating batch %d/%d' % (i + 1, n_batches),
                  end='', flush=True)
        start = i * batch_size
        end = start + batch_size
        images = []
        for f in files[start:end]:
            image = imread(str(f)).astype(np.float32)
            if str(f)[-3:]=='jpg':
                image /= 255
            images.append(image)
        images = np.array(images)
        images = images[:,:,:,0:3]
        images = images.transpose((0, 3, 1, 2))
        #images = images[0,:,:,:]
        batch = torch.from_numpy(images).type(torch.FloatTensor)
        batch = batch.to(torch.device("cuda:0"))
        pred = model(batch)[0]
        pred_arr = pred.cpu().data.numpy().transpose(0, 2, 3, 1).reshape(batch_size*pred.shape[2]*pred.shape[3],-1)
    if verbose:
        print(' done')

    return pred_arr


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)

def calculate_activation_statistics(files, model, batch_size=1,
                                    dims=64, verbose=False):
    act = get_activations(files, model, batch_size, dims, verbose)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma

def calculate_sifid_given_paths(img_path, dir2gen, batch_size, dims, suffix):
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx])
    model = model.to(torch.device("cuda:0"))

    dir2gen = pathlib.Path(dir2gen)
    files = list(dir2gen.glob('*.%s' %suffix))

    fid_values = []
    Im_ind = []
    m1, s1 = calculate_activation_statistics([img_path], model, batch_size, dims)
    for i in range(len(files)):
        m2, s2 = calculate_activation_statistics([files[i]], model, batch_size, dims)
        fid_values.append(calculate_frechet_distance(m1, s1, m2, s2))
    fid_values = np.asarray(fid_values, dtype=np.float32)
    return fid_values.mean()

def calculate_cs(dir2gen, suffix):
    dir2gen = pathlib.Path(dir2gen)
    files = list(dir2gen.glob('*.%s' %suffix))
    images = []
    cvs = 0
    for f in files:
        image = imread(str(f)).astype(np.float64)
        image = image.reshape((image.shape[0]*image.shape[1],-1))
        if suffix=='jpg':
            image /= 255
        images.append(image)
    images = np.array(images)
    images = images[:,:,0:3]
    for i in range(3):
        image_s = images[:,:,i]
        mean_s = np.mean(image_s, axis=0, dtype=np.float64)
        std_s = np.std(image_s, axis=0, dtype=np.float64)
        cv_s = std_s / mean_s
        for j in np.where(np.isnan(cv_s))[0]:
            cv_s[j] = 0
        cvs += np.mean(cv_s, dtype=np.float64)
    return cvs/3


class InceptionV3(nn.Module):
    DEFAULT_BLOCK_INDEX = 3
    BLOCK_INDEX_BY_DIM = {
        64: 0,   # First max pooling features
        192: 1,  # Second max pooling featurs
        768: 2,  # Pre-aux classifier features
        2048: 3  # Final average pooling features
    }
    def __init__(self,
                 output_blocks=[DEFAULT_BLOCK_INDEX],
                 resize_input=False,
                 normalize_input=True,
                 requires_grad=False):
        super(InceptionV3, self).__init__()

        self.resize_input = resize_input
        self.normalize_input = normalize_input
        self.output_blocks = sorted(output_blocks)
        self.last_needed_block = max(output_blocks)
        assert self.last_needed_block <= 3
        self.blocks = nn.ModuleList()
        inception = models.inception_v3(pretrained=True)
        # Block 0: input to maxpool1
        block0 = [
            inception.Conv2d_1a_3x3,
            inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            ]
        self.blocks.append(nn.Sequential(*block0))
        # Block 1: maxpool1 to maxpool2
        if self.last_needed_block >= 1:
            block1 = [
                nn.MaxPool2d(kernel_size=3, stride=2),
                inception.Conv2d_3b_1x1,
                inception.Conv2d_4a_3x3,
            ]
            self.blocks.append(nn.Sequential(*block1))
        # Block 2: maxpool2 to aux classifier
        if self.last_needed_block >= 2:
            block2 = [
                nn.MaxPool2d(kernel_size=3, stride=2),
                inception.Mixed_5b,
                inception.Mixed_5c,
                inception.Mixed_5d,
                inception.Mixed_6a,
                inception.Mixed_6b,
                inception.Mixed_6c,
                inception.Mixed_6d,
                inception.Mixed_6e,
            ]
            self.blocks.append(nn.Sequential(*block2))
        # Block 3: aux classifier to final avgpool
        if self.last_needed_block >= 3:
            block3 = [
                inception.Mixed_7a,
                inception.Mixed_7b,
                inception.Mixed_7c,
            ]
            self.blocks.append(nn.Sequential(*block3))
        if self.last_needed_block >= 4:
            block4 = [
                nn.AdaptiveAvgPool2d(output_size=(1, 1))
            ]
            self.blocks.append(nn.Sequential(*block4))

        for param in self.parameters():
            param.requires_grad = requires_grad

    def forward(self, inp):
        outp = []
        x = inp
        if self.resize_input:
            x = F.upsample(x,
                              size=(299, 299),
                              mode='bilinear',
                              align_corners=False)
        if self.normalize_input:
            x = 2 * x - 1  # Scale from range (0, 1) to range (-1, 1)
        for idx, block in enumerate(self.blocks):
            x = block(x)
            if idx in self.output_blocks:
                outp.append(x)
            if idx == self.last_needed_block:
                break
        return outp