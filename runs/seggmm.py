# -*- coding: utf-8 -*-
"""
File: seggmm.py
Project: MyCPD
Author: Jiachen Zhao
Date: 9/28/19
Description:
"""

import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from gmm import Gmm

X, y = make_blobs(n_features=2, n_samples=1000, centers=3, shuffle=False,random_state=100)   # 500 positive and 500 negative
# plt.scatter(X[:, 0], X[:, 1], marker='o', c=y, s=25, edgecolor='k')
# plt.show()

model = Gmm(n_gmm=2, input_dim=2, latent_dim=2)
y_pred = model.fit_predict(X)
evaluation = cluster_evaluate(y_pred = y_pred, y_true = y_test.flatten())
print(evaluation)



