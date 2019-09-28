# -*- coding: utf-8 -*-
"""
File: gmm.py
Project: MyCPD
Author: Jiachen Zhao
Date: 9/23/19
Description: Achieve the gaussian mixture model with pytorch
"""

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import scipy.io as sio
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import adjusted_mutual_info_score, consensus_score

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

class Gmm(nn.Module):
    def __init__(self, n_gmm = 2,input_dim=2, latent_dim=2):
        super(Gmm, self).__init__()

        # nn module
        layers = []
        # layers += [nn.Linear(input_dim, latent_dim)]
        # layers += [nn.ReLU()]
        layers += [nn.Linear(input_dim, n_gmm)]
        layers += [nn.Softmax(dim=1)]
        self.net = nn.Sequential(*layers)

        # optimizer module
        self.lr = 0.1###################################
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)

        # training process
        self.MAX_ITER = 500#############################
        self.lambda_energy = 1##########################
        self.lambda_cov_diag = 0.001####################

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

    def loss_function(self, z, gamma):
        """
        :param z: original data
        :param gamma: produced by nn
        :param lambda_energy: weight of the energy item in the loss function
        :param lambda_cov_diag: weight of the pan
        :return:
        """
        phi, mu, cov = self.compute_gmm_params(z, gamma)
        sample_energy, cov_diag = self.compute_energy(z, phi, mu, cov, size_average=True)
        loss = self.lambda_energy * sample_energy + self.lambda_cov_diag * cov_diag
        return loss

    def fit(self, X, verbose=False):
        """
        :param X: [num_samples, num_dims], ndarray
        :param MAX_ITER:
        :return:
        """
        X = Variable(torch.from_numpy(X).float())

        for i in range(self.MAX_ITER):
            self.train()
            gamma = self.forward(X)
            self.optimizer.zero_grad()
            loss = self.loss_function(X, gamma)
            loss.backward()
            self.optimizer.step()
            if verbose==True:
                print("Loss is:{:.8f}".format(loss))

    def predict(self, X, y_true):
        """
        :param X: [num_samples, num_dims], ndarray
        :param y_true: flatten truth label vector
        :return:
        """
        X = Variable(torch.from_numpy(X).float())
        gamma_pred = gmm.forward(X)
        _, y_pred = torch.max(gamma_pred, 1)
        y_pred = y_pred.numpy().flatten()
        return y_pred

    def fit_predict(self,X, verbose=False):
        X = Variable(torch.from_numpy(X).float())
        for i in range(self.MAX_ITER):
            self.train()
            gamma = self.forward(X)
            self.optimizer.zero_grad()
            loss = self.loss_function(X, gamma)
            loss.backward()
            self.optimizer.step()
            if verbose == True:
                print("Loss is:{:.8f}".format(loss))
        gamma_pred = self.forward(X)
        _, y_pred = torch.max(gamma_pred, 1)
        y_pred = y_pred.numpy().flatten()
        return y_pred

def cluster_evaluate(y_pred, y_true):
    Acc = np.mean(y_pred == y_true)
    ARI = adjusted_rand_score(y_true, y_pred)
    AMI = adjusted_mutual_info_score(y_true, y_pred)
    evaluation = {'ACC': Acc, 'ARI': ARI, 'AMI': AMI}
    return evaluation


if __name__ == "__main__":
    data_train = sio.loadmat('./data/training.mat')
    data_test = sio.loadmat('./data/testing.mat')
    X_train, y_train = data_train['X'].astype(float), data_train['y']
    X_test, y_test = data_test['X'].astype(float), data_test['y']

    gmm = Gmm(n_gmm=2, input_dim=2, latent_dim=2)
    y_pred = gmm.fit_predict(X_test)
    evaluation = cluster_evaluate(y_pred = y_pred, y_true = y_test.flatten())
    print(evaluation)
