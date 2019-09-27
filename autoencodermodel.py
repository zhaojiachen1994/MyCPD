# -*- coding: utf-8 -*-
"""
File: autoencodermodel.py
Project: MyCPD
Author: Jiachen Zhao
Date: 9/19/19
Description: build a encoder-decoder model to detect the change point
Just compute the reconstructed error without pointing out which is change points
"""

import numpy as np
import pandas as pd
import scipy.io as sio
import torch
from torch import nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from seglearn.transform import SegmentX
# input of SegmentX should be list [array]
# shape of array should be [length, dimension]
# shape of the output is [num_sample, length of sub-sequence, num_dimension]
import ruptures as rpt
# tpt package: http://ctruong.perso.math.cnrs.fr/ruptures-docs/build/html/metrics/display.html

class AutoEncoder(torch.nn.Module):
    def __init__(self, input_dim = 60, latent_dim = 3, output_dim = 60):
        super(AutoEncoder, self).__init__()
        self.lr = 0.01
        self.iterations = 1000
        self.verbose = False

        layers = []
        layers += [nn.Linear(input_dim, 5)]
        layers += [nn.ReLU()]
        layers += [nn.Linear(5,3)]
        layers += [nn.ReLU()]
        layers += [nn.Linear(3,latent_dim)]

        self.encoder = nn.Sequential(*layers)

        layers = []
        layers += [nn.Linear(latent_dim, 3)]
        layers += [nn.ReLU()]
        layers += [nn.Linear(3, 5)]
        layers += [nn.ReLU()]
        layers += [nn.Linear(5, output_dim)]
        self.decoder = nn.Sequential(*layers)


    def forward(self, x):
        """
        :param x: a batch of inputs, x shape is [batch size, num_dimension]
        :return:
        """
        enc = self.encoder(x)
        dec = self.decoder(enc)
        return enc, dec

    def fit(self, x):
        cost = nn.MSELoss(reduction='mean')  # reduction='none'
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        for iter in range(self.iterations):
            _, pred = self.forward(x)
            loss = cost(pred,x)
            if self.verbose == True:
                if iter % 100 ==99:
                    print(iter, loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def predict(self, x):
        enc, dec = self.forward(x)
        return enc, dec

    def fit_predict(self, x):
        self.fit(x)
        enc, dec = self.predict(x)
        return enc, dec

# preprocess data
idx = 2
# data = sio.loadmat('./data/shiftmean%d'%idx) # poor results
# data = sio.loadmat('./data/shiftcorr%d'%idx) # poor results
# data = sio.loadmat('./data/shiftlinear%d'%idx) # poor results
# data = sio.loadmat('./data/singledimshiftfreq%d'%idx)# poor results
# data = sio.loadmat('./data/agotsshiftmean%d'%idx)
# data = sio.loadmat('./data/agotsshiftvar%d'%idx)
data = sio.loadmat('./data/extreme%d'%idx)# good esults
ts = data['ts']#.T # transpose is needed for shiftfreq
print(ts.shape)
bkps = data['bkps'][0]

scaler = StandardScaler()
ts = scaler.fit_transform(ts)

width = 10
step = 5
ts = [ts]
segment = SegmentX(width=width, step=step)
x = segment.fit_transform(ts,None)[0]
x = x.reshape([x.shape[0],-1])
x = torch.from_numpy(x).float()
bkss = bkps//5  #bkss for break samples


model = AutoEncoder(input_dim=10, latent_dim=1, output_dim=10)


_, pred = model.fit_predict(x)

err = (pred-x).detach().numpy()
err = np.max(np.power(err, 2), axis=1)
rpt.display(err,true_chg_pts=bkss)
rpt.display(ts[0],true_chg_pts=bkps)
plt.show()