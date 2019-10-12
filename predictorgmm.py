# -*- coding: utf-8 -*-
"""
File: predictorgmm.py
Project: MyCPD
Author: Jiachen Zhao
Date: 10/12/19
Description:
"""

import numpy as np
import torchvision
import sys
import random
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from seglearn.transform import SegmentX, SegmentXY
import pandas as pd

def to_var(x, volatile=False):
    """
    :param x:
    :param volatile:
    :return:
    Description: tensor or ndarray to variable
    """
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)

class Cholesky(torch.autograd.Function):
    def forward(ctx, a):
        l = torch.cholesky(a, False)
        ctx.save_for_backward(l)
        return l
    def backward(ctx, grad_output):
        l, = ctx.saved_variables
        linv = l.inverse()
        inner = torch.tril(torch.mm(l.t(), grad_output)) * torch.tril(1.0 - Variable(l.data.new(l.size(1)).fill_(0.5).diag()))
        s = torch.mm(linv.t(), torch.mm(inner, linv))
        return s


class SegGmm3(nn.Module):
    def __init__(self):
        super(SegGmm3, self).__init__()

        # PARAMETERS FOR NN STRUCTURE
        self.input_dim = 120
        self.latent_dim = 3
        self.num_gmm = 3
        self.estimation_dim = self.latent_dim + 4

        # training process
        self.lr = 0.01   ##################################
        self.MAX_ITER = 500    #############################
        self.lambda_energy = 0.1    #######################
        self.lambda_cov_diag = 0.00001  ###################
        self.lambda_err_t1 = 1  ###########################

        # DEFINE THE NN STRUCTURE AND OPTIMIZER
        self.buildNN()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

    def buildNN(self):

        # ENCODER
        layers = []
        layers += [nn.Linear(self.input_dim, self.latent_dim)]
        self.encoder = nn.Sequential(*layers)

        # DECODER
        layers = []
        layers += [nn.Linear(self.latent_dim, self.input_dim)]
        self.decoder = nn.Sequential(*layers)

        # PREDICTOR
        layers = []
        layers += [nn.Linear(self.latent_dim, self.input_dim)]
        self.predictor = nn.Sequential(*layers)

        # ESTIMATION
        layers = []
        layers += [nn.Linear(self.estimation_dim, self.num_gmm)]
        layers += [nn.Softmax(dim=1)]
        self.estimator = nn.Sequential(*layers)

    def relative_euclidean_distance(self, a, b):
        return (a-b).norm(2, dim=1) / a.norm(2, dim=1)

    def forward(self, X_t0, X_t1):
        enc = self.encoder(X_t0)    # enc shape is [num_samples, latent_dim]
        dec = self.decoder(enc)     # dec shape is [num_samples, input_dim]
        pre = self.predictor(enc)   # pre shape is [num_samples, input_dim]
        rec_cos_t0 = F.cosine_similarity(X_t0, dec, dim=1) # [num_samples]
        rec_euc_t0 = self.relative_euclidean_distance(X_t0, dec)    # [num_samples]

        rec_cos_t1 = F.cosine_similarity(X_t1, pre, dim=1)
        rec_euc_t1 = self.relative_euclidean_distance(X_t1, pre)
        z = torch.cat([enc, rec_cos_t0.unsqueeze(-1), rec_euc_t0.unsqueeze(-1),
                       rec_cos_t1.unsqueeze(-1), rec_euc_t1.unsqueeze(-1)], dim=1)
        gamma = self.estimator(z)
        return enc, dec, pre, z, gamma

    def compute_gmm_params(self, z, gamma):
        """
        :param z: sample matrix, shape is [num_samples, num_dim]
        :param gamma: the soft mixture-component membership prediction, shape is [num_samples, num_components]
        :return:
        """
        num_samples = z.size(0)
        sum_gamma = torch.sum(gamma, dim=0)
        phi = sum_gamma / num_samples
        # print('phi:', phi)
        self.phi = phi.data
        mu = torch.sum(gamma.unsqueeze(-1) * z.unsqueeze(1), dim=0) / sum_gamma.unsqueeze(-1)
        self.mu = mu.data
        z_minusMu = (z.unsqueeze(1) - mu.unsqueeze(0))  # shape of z_minusMu is [num_samples, num_components, num_dim]
        z_minusMu_outer = z_minusMu.unsqueeze(-1) * z_minusMu.unsqueeze(
            -2)  # shape is [num_samples, num_components, num_dim, num_dim]
        cov = torch.sum(gamma.unsqueeze(-1).unsqueeze(-1) * z_minusMu_outer, dim=0) \
              / sum_gamma.unsqueeze(-1).unsqueeze(-1)  # shape of cov is [num_component, num_dim, num_dim]
        self.cov = cov.data
        return phi, mu, cov

    def compute_energy(self, z, phi=None, mu=None, cov=None, size_average=True):
        """
        :param z: the original data [num_samples, num_dims]
        :param phi: the priori probability of each component
        :param mu: the mean of each component
        :param cov: the covariance matrix of each component
        :param size_average:
        :return: sample_energy
                 cov_diag: the penalty item of the covariance matrix
        """
        if phi is None:
            phi = to_var(self.phi)
        if mu is None:
            mu = to_var(self.mu)
        if cov is None:
            cov = to_var(self.cov)
        # print('phi:', phi)
        num_components, num_dim, _ = cov.size()
        z_minusMu = z.unsqueeze(1) - mu.unsqueeze(0)
        # print(z_minusMu)
        cov_inverse = []  # the covariance list for components
        det_cov = []
        cov_diag = 0
        eps = 1e-3
        for i in range(num_components):
            cov_k = cov[i] + Variable(torch.eye(num_dim) * eps)
            cov_inverse.append(torch.inverse(cov_k).unsqueeze(0))

            eigvals = np.linalg.eigvals(cov_k.data.cpu().numpy() * (2 * np.pi))
            if np.min(eigvals) < 0:
                print(f'Determinant was negative! Clipping Eigenvalues to 0+epsilon from {np.min(eigvals)}')

            # det_cov.append((Cholesky.apply(cov_k * (2 * np.pi)).diag().prod()).unsqueeze(0))

            cov_det_k = torch.cholesky(cov_k * 2 * np.pi, upper=True).diag().prod().unsqueeze(0)
            det_cov.append(cov_det_k)

            cov_diag = cov_diag + torch.sum(1 / cov_k.diag())

        cov_inverse = torch.cat(cov_inverse, dim=0)  # [num_component, num_dim, num_dim]
        det_cov = torch.cat(det_cov)  # [K]
        exp_term_tmp = -0.5 * torch.sum(
            torch.sum(z_minusMu.unsqueeze(-1) * cov_inverse.unsqueeze(0), dim=-2) * z_minusMu, dim=-1)
        max_val = torch.max((exp_term_tmp).clamp(min=0), dim=1, keepdim=True)[0]
        exp_term = torch.exp(exp_term_tmp - max_val)
        sample_energy = -max_val.squeeze() - torch.log(
            torch.sum(phi.unsqueeze(0) * exp_term / (torch.sqrt(det_cov)).unsqueeze(0), dim=1) + eps)

        if size_average == True:
            sample_energy = torch.mean(sample_energy)
        return sample_energy, cov_diag

    def loss_function(self, x_t0, x_hat_t0, x_t1, x_hat_t1, z, gamma):
        recon_error_t0 = torch.mean((x_t0 - x_hat_t0)**2)
        recon_error_t1 = torch.mean((x_t1 - x_hat_t1)**2)
        phi, mu, cov = self.compute_gmm_params(z, gamma)
        sample_energy, cov_diag = self.compute_energy(z, phi, mu, cov)
        loss = recon_error_t0 \
               + self.lambda_err_t1 * recon_error_t1 \
               + self.lambda_energy * sample_energy \
               + self.lambda_cov_diag * cov_diag
        # recon_error_t1, sample_energy, cov_diag = 0, 0, 0
        return loss, recon_error_t0, recon_error_t1, sample_energy, cov_diag

    def fit(self, X_t0, X_t1, verbose=0):
        for iter in range(self.MAX_ITER):
            self.train()
            enc, dec, pre, z, gamma = self.forward(X_t0, X_t1)
            loss, recon_error_t0, recon_error_t1, sample_energy, cov_diag = self.loss_function(X_t0, dec, X_t1, pre, z, gamma)
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 5)
            self.optimizer.step()
            if verbose==1:
                print("Iter{:d}\tloss:{:3.4f}".format(iter, loss))
            elif verbose==2:
                print("Iter{:d}\tloss:{:3.4f}\t err_t0:{:3.4f} \t L*err_t1:{:3.4f}\tL*energy:{:3.4f}\tL*cov_diag:{:3.4f}"
                      .format(iter, loss, recon_error_t0, self.lambda_err_t1*recon_error_t1,
                              self.lambda_energy*sample_energy, self.lambda_cov_diag*cov_diag))

    def predict(self, X_t0, X_t1, verbose = True):
        if torch.is_tensor(X_t0)==False:
            X_t0 = Variable(torch.from_numpy(X_t0).float())
            X_t1 = Variable(torch.from_numpy(X_t1).float())
        enc, dec, pre, z, gamma = self.forward(X_t0, X_t1)
        _, y_pred = torch.max(gamma, 1)
        y_pred = y_pred.numpy().flatten()

        if verbose == True:
            loss, recon_error_t0, recon_error_t1, sample_energy, cov_diag = self.loss_function(X_t0, dec, X_t1, pre, z, gamma)
            print("loss:{:3.4f}\t err_t0:{:3.4f} \t L*err_t1:{:3.4f}\tL*energy:{:3.4f}\tL*cov_diag:{:3.4f}"
                  .format(loss, recon_error_t0, self.lambda_err_t1 * recon_error_t1,
                          self.lambda_energy * sample_energy, self.lambda_cov_diag * cov_diag))
        return y_pred, enc, dec, pre, z, gamma


def demoDataset():
    X, y = make_blobs(n_features=10, n_samples=20000, centers=3, shuffle=False,
                      random_state=1)
    scaler = StandardScaler()
    ts = scaler.fit_transform(X)
    width = 12
    ts = [ts]
    segment = SegmentXY(width=width, overlap=0)#, y_func='middle'
    X, y, _ = segment.fit_transform(ts, [y])#,[y.reshape([-1,1])]
    X = X.reshape(X.shape[0],-1)
    X = Variable(torch.from_numpy(X).float())
    return X, y

def buildOneStepPred(X, y):
    X_t0 = X[:-1, :]
    X_t1 = X[1:,:]
    y = y[:-1]
    data = {'X_t0': X_t0, 'X_t1': X_t1, 'y':y}
    return X_t0, X_t1, y


if __name__ == "__main__":
    X, y = demoDataset()
    X_t0, X_t1, y = buildOneStepPred(X, y)
    seggmm = SegGmm3()
    seggmm.fit(X_t0, X_t1, verbose = 2)
    y_pred, enc, dec, pre, z, gamma = seggmm.predict(X_t0, X_t1, verbose=True)
    print(y_pred)
    z = z.detach().numpy()
    plt.figure(figsize=(10,10))
    plt.subplot(6, 1, 1)
    plt.plot(z[:, 0])
    plt.title('enc')
    plt.subplot(6, 1, 2)
    plt.plot(z[:, 1])
    plt.title('rec_cos_t0')
    plt.subplot(6, 1, 3)
    plt.plot(z[:, 2])
    plt.title('rec_euc_t0')
    plt.subplot(6,1,4)
    plt.plot(z[:, 3])
    plt.title('rec_cos_t1')
    plt.subplot(6,1,5)
    plt.plot(z[:, 4])
    plt.title('rec_euc_t1')
    plt.tight_layout()
    plt.show()