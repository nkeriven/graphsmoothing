#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 10:58:21 2022

@author: kerivenn
"""

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import graph_nk.random as grand
import graph_nk.plot as gplot

from sklearn.kernel_ridge import KernelRidge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression

from scipy.spatial.distance import cdist, pdist, squareform

import torch
import torch.nn as nn
import torch.optim as optim

import gif

plt.close('all')

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


#%% Regression with random projection

np.random.seed(0)
n = 1000
obs = .5
p = 6
small_p = 2
ks = np.arange(1,100)

X = np.random.rand(n,p)*(1+np.random.rand(p)[None,:])/2
bandwidth = np.median(pdist(X, 'euclidean'))/3
true_omega = np.random.randn(p,small_p)
true_omega /= np.sqrt((true_omega**2).sum(axis=0)[None,:])
X_obs = X @ true_omega
true_beta = np.random.randn(p,1)
true_beta /= np.linalg.norm(true_beta)
sigma = .01
y = X @ true_beta + sigma*np.random.randn(n,1)
vary = np.mean((y-np.mean(y))**2)

G, W = grand.random_graph_similarity(X, return_expected=True, mode='Gaussian',
                                     bandwidth=bandwidth)

W = W+ np.eye(n)
D = W.sum(axis=0)
W = W/D[:,None]
# W /= n

model = LinearRegression()

mse=[]

obs_ind = np.random.choice(np.arange(n), size=int(n*obs), replace=False)
obs_mask = np.zeros(n)
obs_mask[obs_ind] = 1
test_ind = np.where(1-obs_mask)[0]

X_smooth = X_obs.copy()
for k in ks:
    model.fit(X_smooth[obs_ind,:], y[obs_ind])
    yhat = model.predict(X_smooth[test_ind,:])
    mse.append(((yhat.squeeze() - y[test_ind,:].squeeze())**2).sum()/len(test_ind))
    X_smooth = W@X_smooth

plt.figure()
plt.plot(ks,mse)
plt.plot(ks, vary*np.ones_like(ks))

#%%


#%% Gif

def MSE(X, y):
    return np.linalg.lstsq(X, y)[1]

@gif.frame
def plot_graph(G,W, X, color, mse, xlim=None, ylim=None):
    fig, axs = plt.subplots(1, 2, figsize=(20,10))

    gplot.my_draw(G, pos=X, node_color=color, width=[W[i,j] for (i,j) in G.edges()], ax = axs[0])
    # _ = ax.axis('off')
    axs[0].set_xlim(xlim)
    axs[0].set_ylim(ylim)

    axs[1].plot(mse)
    axs[1].set_title('MSE')
    # if title is not None:
    #     plt.title(title)

def animate(G,W, X_start, X_end, color, mse, num_steps=20, xlim=None, ylim=None):
    ts = np.linspace(0,1,num_steps)
    frames = []
    for i in range(num_steps-1):
        t = ts[i]
        pos = (1-t)*X_start + t*X_end
        frames.append(plot_graph(G,W, pos, color, mse, xlim=xlim, ylim=ylim))
    return frames

frames = []
mse=[]
xlim = [X_obs[:,0].min(), X_obs[:,0].max()]
ylim = [X_obs[:,1].min(), X_obs[:,1].max()]
X_smooth = X_obs.copy()
for k in range(100):
    model.fit(X_smooth, y)
    yhat = model.predict(X_smooth)
    mse.append(((yhat.squeeze() - y.squeeze())**2).sum()/n)
    title = f'k = {k}, mse = {mse[-1]}'
    X_next = W@X_smooth
    frames.extend(animate(G, .5*W/W.max(), X_smooth, X_next, y, mse, num_steps=4))
    X_smooth = X_next

gif.save(frames, 'gml/smoothing.gif', duration=50)

#%%

