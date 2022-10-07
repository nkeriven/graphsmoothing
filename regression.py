
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import pdist, squareform
import scipy


plt.close('all')
np.random.seed(0)
savefig=False

#%%

d = 2
p = 1
lbd = .001 # regularization parameter
kmax = 10
# eigenvalues

if False: # switch to False for failure of beneficial smoothing
    lamb1 = 2
    lamb2 = .5
    n = 1000
else:
    lamb1 = .5
    lamb2 = 1
    n = 3000

X = np.random.randn(n, d)
u1 = np.array([[1],[1]])/np.sqrt(2)
u2 = np.array([[-1],[1]])/np.sqrt(2)
Sigma = lamb1*u1@u1.T + lamb2*u2@u2.T
betastar = u1
X = X@ scipy.linalg.sqrtm(Sigma)
Y = X@betastar
M = np.eye(d,p)

#%% smoothing
W = np.exp(-squareform(pdist(X, 'sqeuclidean'))/2)
D = W.sum(axis=0)
W = W/D[:,None]

WX = X

acc=[]
th_acc = []
for k in range(kmax):
    WXM = WX @M

    beta = np.linalg.solve(WXM.T @ WXM/n + lbd*np.eye(p), WXM.T @ Y/n)
    yhat = WXM @ beta


    if k<3:
        plt.figure(figsize=(6,3.5))
        plt.scatter(WX[:,0], WX[:,1], c=Y)
        plt.xlim([-2.5, 2.5])
        plt.ylim([-3.5, 3.5])
        if savefig:
            plt.savefig(f'reg_latent_{k}.pdf',
                        bbox_inches='tight', transparent=True)
        plt.figure(figsize=(6,2.4))
        ind = np.argsort(WX[:,0])
        plt.plot(WX[ind,0], Y[ind])
        plt.xlim([-3, 3])
        plt.xlabel('z', fontsize=16)
        plt.ylabel('y', fontsize=16)
        if savefig:
            plt.savefig(f'reg_features_{k}.pdf',
                        bbox_inches='tight', transparent=True)

    acc.append(np.linalg.norm(Y-yhat)**2/(n))

    lamb1mod = lamb1/(1+1/lamb1)**(2*k)
    lamb2mod = lamb2/(1+1/lamb2)**(2*k)
    th_acc.append(lamb1*((2*lbd+lamb2mod)**2 + lamb1mod*lamb2mod)/(2*lbd+lamb1mod+lamb2mod)**2)


    WX=W@WX

plt.figure(figsize=(5,4))
plt.semilogx(np.arange(1, kmax+1), acc, label='Empirical', linewidth=3)
plt.semilogx(np.arange(1, kmax+1), th_acc, label='Theory', linewidth=3)
plt.xlabel('Order of smoothing', fontsize=12)
plt.ylabel('MSE', fontsize=12)
plt.legend(fontsize=14)
if savefig:
    plt.savefig('reg_MSE.pdf',
                bbox_inches='tight', transparent=True)
