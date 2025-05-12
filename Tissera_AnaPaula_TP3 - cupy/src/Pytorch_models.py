

import torch
import torch.nn as nn
import torch.optim as optim
import cupy as cp


class MLP_M2(nn.Module):
    def __init__(self, conf):
        super().__init__()
        layers = []
        sizes  = conf['arch']
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i+1]))
            if i < len(sizes) - 2:
                if conf['use_batchnorm']:
                    layers.append(nn.BatchNorm1d(sizes[i+1]))
                layers.append(nn.ReLU())
                if conf['dropout_p'] > 0:
                    layers.append(nn.Dropout(conf['dropout_p']))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
    
    
class MLP_M3(nn.Module):
    def __init__(self, conf):
        super().__init__()
        layers = []
        sizes  = conf['arch']
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i+1]))
            if i < len(sizes) - 2:
                if conf['use_batchnorm']:
                    layers.append(nn.BatchNorm1d(sizes[i+1]))
                layers.append(nn.ReLU())
                if conf['dropout_p'] > 0:
                    layers.append(nn.Dropout(conf['dropout_p']))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)