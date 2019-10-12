# -*- coding: utf-8 -*-
"""
File: seggmm.py
Project: MyCPD
Author: Jiachen Zhao
Date: 9/28/19
Description:对dagmm的复现
"""

import numpy as np
from sklearn.datasets import make_blobs
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

from gmm import Gmm,cluster_evaluate

X, y = make_blobs(n_features=2, n_samples=1000, centers=3, shuffle=False,random_state=100)   # 500 positive and 500 negative
# plt.scatter(X[:, 0], X[:, 1], marker='o', c=y, s=25, edgecolor='k')
# plt.show()

# model = Gmm(n_gmm=3, input_dim=2, latent_dim=1)
# y_pred = model.fit_predict(X)
# evaluation = cluster_evaluate(y_pred = y_pred, y_true = y.flatten())
# print(evaluation)

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

class SegGmm(nn.Module):
    def __init__(self, input_dim=2, latent_dim=1, estimation_dim = 3, output_dim=2):
        """
        :param input_dim: dimension of original data
        :param latent_dim: dimension of encoder output
        :param estimation_dim: dimension of estimation input
        :param output_dim: dimension of estimation output
        """
        super(SegGmm, self).__init__()
        # nn module
        layers = []
        layers += [nn.Linear(input_dim, latent_dim)]
        self.encoder = nn.Sequential(*layers)

        layers = []
        layers += [nn.Linear(latent_dim, input_dim)]
        self.decoder = nn.Sequential(*layers)

        layers = []
        layers += [nn.Linear(estimation_dim, 10)]
        layers += [nn.Tanh()]
        layers += [nn.Dropout(p=0.5)]
        layers += [nn.Linear(10,output_dim)]#
        layers += [nn.Softmax(dim=1)]

        self.estimation = nn.Sequential(*layers)

        # optimizer module
        self.lr = 0.01###################################
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        # training process
        self.MAX_ITER = 50#############################
        self.lambda_energy = 0.1##########################
        self.lambda_cov_diag = 0.00001####################

        self.PARAMS={}

    def relative_euclidean_distance(self, a, b):
        return (a-b).norm(2, dim=1) / a.norm(2, dim=1)

    def forward(self, x):
        enc = self.encoder(x)   # size is [num_samples, latent_dim]
        dec = self.decoder(enc) # size is [num_samples, input_dim]
        rec_cosine = F.cosine_similarity(x, dec, dim=1) # size is [num_samples]
        rec_euclidean = self.relative_euclidean_distance(x, dec)    # size is [num_samples]

        z = torch.cat([enc, rec_euclidean.unsqueeze(-1), rec_cosine.unsqueeze(-1)], dim=1) # size is [num_samples, estimation_dim = 3]
        # z = enc
        gamma = self.estimation(z)


        return enc, dec, z, gamma

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
        # print('phi.data:', self.phi)
        # phi = to_var(self.phi)
        # print('phi:', phi)
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
    # ????? why sample energy could be negative

    def loss_function(self, x, x_hat, z, gamma, lambda_energy, lambda_cov_diag):
        recon_error = torch.mean((x-x_hat)**2)
        phi, mu, cov = self.compute_gmm_params(z, gamma)
        print('phi:', phi)
        sample_energy, cov_diag = self.compute_energy(z, phi, mu, cov)
        loss = recon_error + lambda_energy * sample_energy + lambda_cov_diag * cov_diag
        return loss, recon_error, sample_energy, cov_diag

    def fit(self, X, verbose=False):
        """
        :param X: [num_samples, num_dims], ndarray
        :param verbose:
        :return:
        """
        X = Variable(torch.from_numpy(X).float())

        for iter in range(self.MAX_ITER):
            self.train()
            enc, dec, z, gamma = self.forward(X)
            loss, recon_error, sample_energy, cov_diag = \
                self.loss_function(X, dec, z, gamma, self.lambda_energy, self.lambda_cov_diag)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 5)
            self.optimizer.step()
            if verbose == True:
                print("loss:{:0.4f}, recon_error:{:0.4f}, sample_energy:{:0.4f}, cov_diag:{:0.4f}".format(loss, recon_error, self.lambda_energy * sample_energy, self.lambda_cov_diag * cov_diag))

    def predict(self, X):
        X = Variable(torch.from_numpy(X).float())
        enc, dec, z, gamma = self.forward(X)
        _, y_pred = torch.max(gamma, 1)
        y_pred = y_pred.numpy().flatten()
        return enc, dec, z, gamma, y_pred

    def fit_predict(self, X, verbose= False):
        self.fit(X, verbose=verbose)
        y_pred = self.predict(X)
        return y_pred

if __name__ == "__main__":
    X, y = make_blobs(n_features=5, n_samples=300, centers=3, shuffle=False,
                      random_state=18)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    # X = Variable(torch.from_numpy(X).float())
    model = SegGmm(input_dim=5, latent_dim=3, estimation_dim=5, output_dim=3)
    model.fit(X, verbose=True)
    enc, dec, z, gamma, y_pred = model.predict(X)
    # bkps_pred = label2bkps(y_pred, gap=3, verbose=False)
    # print(bkps_pred)
    # print(y_pred)
    # print(y)
    # enc, dec, z, gamma = model(X)
    # print('enc size:', enc.size())
    # print('dec size:', dec.size())
    # print('z size:', z.size())
    # print('gamma size:', gamma.size())
    # print(gamma)
    # print(np.argmax(gamma.detach().numpy(), axis=1))
    plt.scatter(X[:, 0], X[:, 1], marker='o', c=y,
                    s=25, edgecolor='k')
    plt.figure()
    plt.plot(enc.detach().numpy())

    plt.figure()
    plt.scatter(dec.detach().numpy()[:, 0], dec.detach().numpy()[:, 1], marker='o', c=y,
                    s=25, edgecolor='k')
    # plt.figure()
    # plt.plot(z.detach().numpy()[:,1])
    # plt.figure()
    # plt.plot(z.detach().numpy()[:,2])
    # plt.show()



