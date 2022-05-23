
#%%

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import pdist, squareform
import seaborn as sns

#%%
plt.close('all')
np.random.seed(0)
save = False

#%%
n = 2000
d = 2
p = 1
lbd = .01 # regularization parameter
hardness = .94 # alignement between mu and M. 1 is unsolvable problem.
err_const = .1 # multiplicative constant for the theoretical error
kmax = 100 # order of smoothing

#%%
X = np.zeros((2*n, d))
mu = np.random.randn(d)
if d>p:
    mu *= 1 - .94*np.concatenate((np.ones(p), np.zeros(d-p)))
mu /= np.linalg.norm(mu)
mu *= 3/np.sqrt(d)

X[:n, :] = mu+ np.random.randn(n,d)
X[n:, :] = -mu + np.random.randn(n,d)

Y = np.zeros(2*n)
Y[:n] = 1
Y[n:] = -1

M = np.eye(d,p)
C = np.linalg.norm(mu@M)
print('Norm nu:', C)

XM = X@M

#%%
W = np.exp(-squareform(pdist(X, 'sqeuclidean'))/2)
D = W.sum(axis=0)
W = W/D[:,None]

#%%
WX = X
acc=[]
th_acc = []
err=0
for k in range(kmax):
    WXM=WX@M

    beta = np.linalg.solve(WXM.T @ WXM/(2*n) + lbd*np.eye(p), WXM.T @ Y/(2*n))
    yhat = WXM @ beta

    if k<3:

        plt.figure(figsize=(6,3.5))
        sns.kdeplot(x=WX[:n,0], y=WX[:n,1], fill=True, alpha=.8, levels=20, thresh=.02,)
        sns.kdeplot(x=WX[n:,0], y=WX[n:,1], fill=True, alpha=.8, levels=20, thresh=.02,)
        plt.xlim([-3, 3])
        plt.ylim([-4.5, 4.5])
        if save:
            plt.savefig(f'classif_latent_{k}.pdf',
                        bbox_inches='tight', transparent=True)

        plt.figure(figsize=(6,2.5))
        sns.kdeplot(data=WXM[:n].squeeze(), shade=True, legend=False, color='tab:blue', linewidth=2)
        sns.kdeplot(data=WXM[n:].squeeze(), shade=True, legend=False, color='tab:orange', linewidth=2)
        plt.xlim([-3, 3])
        if save:
            plt.savefig(f'classif_features_{k}.pdf',
                        bbox_inches='tight', transparent=True)

    acc.append(np.linalg.norm(Y-yhat)**2/(2*n))
    c = (1/4)**(k)
    th_acc.append(((c+lbd)**2 + (C**2)*c)/(c+lbd+C**2)**2+err)
    err = err+ err_const*np.exp(-np.linalg.norm(mu)**2/(2+2/4**k))
    WX=W@WX

plt.figure(figsize=(5,4))
plt.semilogx(np.arange(1, kmax+1), acc, label='Empirical', linewidth=3)
plt.semilogx(np.arange(1, kmax+1), th_acc, label='Theory', linewidth=3)
plt.xlabel('Order of smoothing', fontsize=12)
plt.ylabel('MSE', fontsize=12)
plt.legend(fontsize=14)
if save:
    plt.savefig('classif_MSE.pdf',
                bbox_inches='tight', transparent=True)
