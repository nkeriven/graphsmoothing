
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from sklearn.decomposition import PCA
import dgl # for datasets. Can be anything, really

plt.close('all')
np.random.seed(0)
save = False

#%%
kmax = 500
p = 200
lbd = 1e-7 # regularization

g = dgl.data.CoraGraphDataset()[0]
A = g.adj(scipy_fmt='coo')

#extract largest CC
G = nx.from_scipy_sparse_matrix(A)
Gcc = max(nx.connected_components(G), key=len)
Gcc = G.subgraph(Gcc).copy()
ind = [i for i in Gcc.nodes()]
n = len(ind)

W = nx.to_numpy_array(Gcc)
W = W + np.eye(n)
D = W.sum(axis=0)
W = W/D[:,None]

# PCA and center
p = 200
pca = PCA(n_components = p)
X = np.array(g.ndata['feat'])[ind,:]
X = pca.fit_transform(X)
X -= X.mean(axis=0)[None,:]

# label
Y = np.array(g.ndata['label'])[ind]
Y = np.array(Y, dtype=np.float64) - Y.mean()

#%%

WX = X
acc=[]
th_acc = []
for k in range(kmax):

    beta = np.linalg.solve(WX.T @ WX/n + lbd*np.eye(p), WX.T @ Y/n)
    yhat = WX @ beta
    acc.append(np.linalg.norm(Y-yhat)**2/n)

    WX=W@WX
    if k==0 or k==10 or k==499:
        plt.figure(figsize=(6,3))
        ind = [i for i in range(n) if np.abs(Y[i])<1.5]
        plt.scatter(WX[ind,0], WX[ind,1], c=Y[ind], s=15)
        plt.xlim([-.05,.06])
        plt.ylim([-.1,.1])
        plt.xlabel('x0', fontsize=12)
        plt.ylabel('x1', fontsize=12)
        if save:
            plt.savefig(f'oversmoothing/fig/cora_data{k}.pdf',
                        bbox_inches='tight', transparent=True)


plt.figure(figsize=(6,3))
plt.semilogx(np.arange(1, kmax+1), acc, label='Empirical', linewidth=3)
plt.xlabel('Order of smoothing', fontsize=16)
plt.ylabel('MSE', fontsize=16)
# plt.legend(fontsize=14)
if save:
    plt.savefig('oversmoothing/fig/cora_MSE.pdf',
                bbox_inches='tight', transparent=True)
