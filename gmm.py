# -*- coding: utf-8 -*-
"""
File: gmm.py
Project: MyCPD
Author: Jiachen Zhao
Date: 9/23/19
Description:
"""

import numpy as np
import pandas as pd
import torch
import torchvision
import torch.nn as nn
from torch.autograd import Variable
import scipy.io as sio


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

# def compute_gmm_params(z, gamma):
#     """
#     :param z: sample matrix, shape is [num_samples, num_dim]
#     :param gamma: the soft mixture-component membership prediction, shape is [num_samples, num_components]
#     :return: phi: priori probs of each component, dtype = torch.tensor
#              mu: mean of each component, dtype = torch.tensor
#              cov: covariance of each component, dtype = torch.tensor
#     """
#     num_samples = z.size(0)
#     sum_gamma = torch.sum(gamma, dim=0)
#     phi = sum_gamma/num_samples
#
#     mu = torch.sum(gamma.unsqueeze(-1)*z.unsqueeze(1), dim=0) / sum_gamma.unsqueeze(-1)
#     z_minusMu = (z.unsqueeze(1) - mu.unsqueeze(0)) # shape of z_minusMu is [num_samples, num_components, num_dim]
#     z_minusMu_outer = z_minusMu.unsqueeze(-1) * z_minusMu.unsqueeze(-2) # shape is [num_samples, num_components, num_dim, num_dim]
#     cov = torch.sum(gamma.unsqueeze(-1).unsqueeze(-1) * z_minusMu_outer, dim = 0)\
#           / sum_gamma.unsqueeze(-1).unsqueeze(-1)# shape of cov is [num_component, num_dim, num_dim]
#     return phi, mu, cov

# def compute_energy(z, phi, mu, cov, size_average = True):
#     """
#     :param z: the original data [num_samples, num_dims]
#     :param phi: the priori probability of each component
#     :param mu: the mean of each component
#     :param cov: the covariance matrix of each component
#     :param size_average:
#     :return: sample_energy
#              cov_diag: the penalty item of the covariance matrix
#     """
#     # compute the equation 6 in the original paper
#     num_components, num_dim, _ = cov.size()
#     z_minusMu = z.unsqueeze(1) - mu.unsqueeze(0)
#     cov_inverse = [] # the covariance list for components
#     det_cov = []
#     cov_diag = 0
#     eps = 1e-12
#     for i in range(num_components):
#         cov_k = cov[i]+Variable(torch.eye(num_dim) * eps)
#         cov_inverse.append(torch.inverse(cov_k).unsqueeze(0))
#         cov_det_k = torch.cholesky(cov_k*2*np.pi, upper=True).diag().prod().unsqueeze(0)
#         det_cov.append(cov_det_k)
#         cov_diag = cov_diag + torch.sum(1 / cov_k.diag())
#
#     cov_inverse = torch.cat(cov_inverse, dim=0) # [num_component, num_dim, num_dim]
#     det_cov = torch.cat(det_cov) # [K]
#     exp_term_tmp = -0.5 * torch.sum(torch.sum(z_minusMu.unsqueeze(-1) * cov_inverse.unsqueeze(0), dim=-2) * z_minusMu, dim=-1)
#     max_val = torch.max((exp_term_tmp).clamp(min=0), dim=1, keepdim=True)[0]
#     exp_term = torch.exp(exp_term_tmp - max_val)
#     sample_energy = -max_val.squeeze() - torch.log(
#         torch.sum(phi.unsqueeze(0) * exp_term / (torch.sqrt(det_cov)).unsqueeze(0), dim=1) + eps)
#
#     if size_average == True:
#         sample_energy = torch.mean(sample_energy)
#     return sample_energy, cov_diag

# def loss_function(z, gamma, lambda_energy, lambda_cov_diag):
#     """
#     Description: use c
#     :param z:
#     :param gamma:
#     :param lambda_energy:
#     :param lambda_cov_diag:
#     :return:
#     """
#     phi, mu, cov = compute_gmm_params(z, gamma)
#     sample_energy, cov_diag = compute_energy(z, phi, mu, cov, size_average=True)
#     loss = lambda_energy*sample_energy + lambda_cov_diag*cov_diag
#     return loss


class Gmm(nn.Module):
    def __init__(self, n_gmm = 2,input_dim=2, latent_dim=2):
        super(Gmm, self).__init__()
        layers = []
        # layers += [nn.Linear(input_dim, latent_dim)]
        # layers += [nn.ReLU()]
        layers += [nn.Linear(input_dim, n_gmm)]
        layers += [nn.Softmax(dim=1)]
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        gamma = self.net(z)
        return gamma

    def compute_gmm_params(self, z, gamma):
        """
        :param z: sample matrix, shape is [num_samples, num_dim]
        :param gamma: the soft mixture-component membership prediction, shape is [num_samples, num_components]
        :return: phi: priori probs of each component, dtype = torch.tensor
                 mu: mean of each component, dtype = torch.tensor
                 cov: covariance of each component, dtype = torch.tensor
        """
        num_samples = z.size(0)
        sum_gamma = torch.sum(gamma, dim=0)
        phi = sum_gamma / num_samples
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
        num_components, num_dim, _ = cov.size()
        z_minusMu = z.unsqueeze(1) - mu.unsqueeze(0)
        cov_inverse = []  # the covariance list for components
        det_cov = []
        cov_diag = 0
        eps = 1e-12
        for i in range(num_components):
            cov_k = cov[i] + Variable(torch.eye(num_dim) * eps)
            cov_inverse.append(torch.inverse(cov_k).unsqueeze(0))
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

    def loss_function(self, z, gamma, lambda_energy=1, lambda_cov_diag=1):
        """

        :param z:
        :param gamma:
        :param lambda_energy: weight of the energy item in the loss function
        :param lambda_cov_diag: weight of the pan
        :return:
        """
        phi, mu, cov = self.compute_gmm_params(z, gamma)
        sample_energy, cov_diag = self.compute_energy(z, phi, mu, cov, size_average=True)
        loss = lambda_energy * sample_energy + lambda_cov_diag * cov_diag
        return loss

MAX_ITER = 500
gmm = Gmm(n_gmm=2, input_dim=2, latent_dim=2)
optimizer = torch.optim.Adam(gmm.parameters(),lr=0.1)

data_train = sio.loadmat('./data/training.mat')
data_test = sio.loadmat('./data/testing.mat')
X_train, y_train = data_train['X'].astype(float), data_train['y']
X_test, y_test = data_test['X'].astype(float), data_test['y']
X_train = torch.from_numpy(X_train).float()
X_train = Variable(X_train)
# y_train = torch.from_numpy(y_train).float()
# X_test = torch.from_numpy(X_test).float()
# y_test = torch.from_numpy(y_test).float()



for i in range(MAX_ITER):
    gmm.train()
    gamma = gmm(X_train)
    optimizer.zero_grad()
    loss = gmm.loss_function(X_train, gamma, lambda_energy=0.5, lambda_cov_diag=0.001)
    loss.backward()
    optimizer.step()
    print(loss)

gamma_pred = gmm(X_train)
_, y_pred = torch.max(gamma_pred,1)
y_pred = y_pred.numpy()
print(y_pred)
print(y_train)
print(np.mean(y_pred==y_train.flatten()))

#


