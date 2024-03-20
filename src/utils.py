import torch
import torch.nn.functional as F
from torch import nn, optim

import numpy as np

from collections import OrderedDict
from types import SimpleNamespace
from time import strftime, localtime
import random
import argparse
import itertools
import json
import os
import sys

import sklearn.decomposition

# ---- convenience functions

def p_samp(p, num_samp, c = None, w = None):
    repflag = p.shape[0] < num_samp
    p_sub = np.random.choice(p.shape[0], size = num_samp, replace = repflag)
    if w is None:
        w_ = torch.ones(len(p_sub))
    else:
        w_ = w[p_sub].clone()
    w_ = w_ / w_.sum()


    if c is None :
      return p[p_sub,:].clone(), w_,
    else:
      return p[p_sub,:].clone(), w_, c[p_sub,:]
    # return p[p_sub,:].clone()


def fit_regularizer(samples, pp, burnin, dt, sd, model, device, y_samples, y):
    factor = samples.shape[0] / pp.shape[0]

    # print(pp.shape)

    z = torch.randn(burnin, pp.shape[0], pp.shape[1]) * sd
    z = z.to(device)

    for i in range(burnin):
        pp,_ = model._step(pp, y, dt, z = z[i,:,:])

    pos_fv = -1 * model._pot(samples, y = y_samples).sum()
    neg_fv = factor * model._pot(pp.detach(), y = y).sum()
    return pp, pos_fv, neg_fv

def get_weight(w, time_elapsed):
    return w

def init(args):

    args.cuda = torch.cuda.is_available()
    device = torch.device('cuda:{}'.format(args.gpu) if args.cuda else 'cpu')
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

    return device, kwargs

def weighted_samp(p, num_samp, w):
    ix = list(torch.utils.data.WeightedRandomSampler(w, num_samp))
    return p[ix,:].clone()

def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")

def load_model(obj, path, epoch="train.best.pt", device='cuda'):
    config_path = os.path.join(path, "config.pt")
    config = SimpleNamespace(**torch.load(config_path))
    model = obj(config)
    train_pt = os.path.join(path, "train.best.pt")
    checkpoint = torch.load(train_pt, map_location = device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    return model

