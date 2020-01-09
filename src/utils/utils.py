import math
from random import randint

import torch
import numpy as np
from scipy.io import loadmat

import constants


def mod_crop(blob, N):
    if isinstance(blob, np.ndarray):
        # For numpy arrays, channels at the last dim
        h, w = blob.shape[-3:-1]
        nh = h - h % N
        nw = w - w % N
        return blob[..., :nh, :nw, :]
    else: 
        # For 4-D pytorch tensors, channels at the 2nd dim
        with torch.no_grad():
            h, w = blob.shape[-2:]
            nh = h - h % N
            nw = w - w % N
            return blob[..., :nh, :nw]


def create_rgb(sens, hsi):
    # If hsi contains a batch dim and sens does not, use this sens for all samples in the mini-batch
    # sens: cx3 or bxcx3
    # hsi:  bxcxhxw
    # rgb:  bx3xhxw
    b, c, h, w = hsi.size()
    return torch.matmul(sens.transpose(-1, -2), hsi.view(b, c, -1)).view(b, c, h, w)

    
# These functions below are adapted from 
# https://github.com/berk95kaya/Spectral-Estimation/blob/3c634de1ba196e3429e299c25bfee35f440c7e27/functions.py
def create_sensitivity(sens_type, idx=None):
    if sens_type == 'D':
        if idx is None:
            idx = randint(0, constants.SENS_NUM-1)
        else:
            idx = int(idx)
        f = loadmat(constants.SENS_FILE)
        f = f['sensitivities']
        sens = f['sens'+str(idx)]
        return torch.from_numpy(sens[0,0]), idx
    elif sens_type == 'C':
        ## Note that this was hard-coded for 31-band case!
        return torch.from_numpy(_create_sens_mixture())
    else:
        raise NotImplementedError("bad sensitivity type")


def _create_sens_gaussian():
    x = np.arange(1,32)

    mean_red = np.random.uniform(low = 16 ,high = 26 )
    mean_green = np.random.uniform(low = 10 ,high = 20 )
    mean_blue = np.random.uniform(low = 5 ,high = 15 )
    
    sigma_red = np.random.uniform(low = 2 ,high = 6 )
    sigma_green = np.random.uniform(low = 2 ,high = 6 )
    sigma_blue = np.random.uniform(low = 2 ,high = 6 )
    
    y_red = np.exp(  -(x-mean_red)**2 / sigma_red**2)
    y_green = np.exp(  -(x-mean_green)**2 / sigma_green**2)
    y_blue = np.exp(  -(x-mean_blue)**2 / sigma_blue**2)
    
    sens = np.transpose(np.concatenate(([y_red],  [y_green], [y_blue]),axis=0))/8
    
    return sens


def _create_sens_mixture():
    k= randint(1,5)
    r = [random.random() for i in range(0,k)]
    r = r/np.sum(r)
    sens = np.zeros([31,3])
    for i in range(k):
        sens = sens+ r[i]*create_sens_gaussian() 
    return sens