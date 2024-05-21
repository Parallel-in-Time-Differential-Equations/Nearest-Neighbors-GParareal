from globals import *

#%% Visualizing the dataset


import time
from configs import Config
from systems import FHN_ODE, Brusselator
from solver import SolverRK
from parareal import PararealLight, Parareal
import numpy as np
import matplotlib.pyplot as plt

import warnings
from scipy.linalg import LinAlgWarning
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=LinAlgWarning)


ode = Brusselator()
config = Config(ode).get()
config['tspan'] = [0,15.5]
config['Ng']=1
config['N'] = 40
s = SolverRK(ode.get_vector_field(), **config)
p  = Parareal(ode, s, **config)

truth = s.run_F_full(*config['tspan'], p.u0)
plt.plot(*truth.T)

#%


out = p.run('nngp', pool=10)

out.keys()
u = out['u']
data_x, data_D = out['data_x'], out['data_D']
k = out['k']
mn = np.nanmin(data_x)
mx = np.nanmax(data_x)
tr = lambda x: 2*(x-mn)/(mx-mn)-1
data_x = tr(data_x)
u = tr(u)
# data_x[:,0,:] /= 2
# u[:,0,:] /= 2
k = 4
i = 25

fs = 14
fig, axs = plt.subplots(1,2, figsize=(11,5))
ax = axs[0]
target = u[30,:,-1]
art1 = ax.scatter(*target, c='blue', s=35)
colors = plt.get_cmap('copper')(np.array(range(data_x.shape[0]))/data_x.shape[0])
for i in range(data_x.shape[0]):
    art2 = ax.scatter(*data_x[i,:,:-1], c='gray', s=3)
ax.set_xlabel('x coordinate')
ax.set_ylabel('y coordinate')
ax.legend([art2, art1], [r'$U \in \mathcal{U}_6$', '$U_{30}^{6}$'], fontsize=14)
    
ax=axs[1]

# fig,ax = plt.subplots()
target = u[30,:,-1]
colors = plt.get_cmap('copper')(np.array(range(data_x.shape[0]))/data_x.shape[0])
for i in range(data_x.shape[0]):
    for k in range(data_x.shape[-1]-1):
        data = np.log10(np.abs(data_x[i,:,k]-target))
        ax.scatter(*data, c='blue', s=5)
        if data[1] < -1.5 and data[0] < -2:
            if data[0] < -5:
                ax.text(data[0]+0.18, data[1]-0.16, '$(\mathscr{F}-\mathscr{G})'+f'(U_{{{i}}}^{{{k}}})$', fontsize=14)
            elif data[0] < -3.2:
                ax.text(data[0]-1.1, data[1]-0.5, '$(\mathscr{F}-\mathscr{G})'+f'(U_{{{i}}}^{{{k}}})$', fontsize=14)
            elif data[1]<-2:
                ax.text(data[0]+0.18, data[1]-0.16, '$(\mathscr{F}-\mathscr{G})'+f'(U_{{{i}}}^{{{k}}})$', fontsize=14)
            else:
                ax.text(data[0]-2.4, data[1]-0.16, '$(\mathscr{F}-\mathscr{G})'+f'(U_{{{i}}}^{{{k}}})$', fontsize=14)
        

data.shape


ax=axs[1]
tmp = np.log10(np.abs(data_x[:,:,:-1]-target.reshape(1,-1,1)))
temp = np.moveaxis(tmp, 0, -1).reshape(2,-1)
dist = (np.exp(temp)**2).sum(0)
mask = np.logical_not(np.isnan(dist))
sorted_idx = np.argsort(dist[mask])
for i in range(15):
    x, y = temp[:,mask][:,sorted_idx][:,i]
    ax.scatter(x, y, c='red', s=5)

ax.set_xlabel('x coordinate (log)') 
ax.set_ylabel('y coordinate (log)') 
ax.set_title('Plot of $|U - U_{30}^{6}|, U \in D_6$')


ax=axs[0]
org = data_x[:,:,:-1]
temp_org = np.moveaxis(org, 0, -1).reshape(2,-1)
tmp = np.log10(np.abs(data_x[:,:,:-1]-target.reshape(1,-1,1)))
temp = np.moveaxis(tmp, 0, -1).reshape(2,-1)
dist = (np.exp(temp)**2).sum(0)
mask = np.logical_not(np.isnan(dist))
sorted_idx = np.argsort(dist[mask])
for i in range(15):
    x, y = temp_org[:,mask][:,sorted_idx][:,i]
    ax.scatter(x, y, c='red', s=5)
fig.tight_layout()


store_fig(fig, 'brus_dataset_vis_para_both')

fig, ax = plt.subplots()
target = u[30,:,-1]
art1 = ax.scatter(*target, c='blue', s=35)
colors = plt.get_cmap('copper')(np.array(range(data_x.shape[0]))/data_x.shape[0])
for i in range(data_x.shape[0]):
    art2 = ax.scatter(*data_x[i,:,:-1], c='gray', s=3)
ax.set_xlabel('x coordinate')
ax.set_ylabel('y coordinate')
ax.legend([art2, art1], [r'$U \in \mathcal{U}_6$', '$U_{30}^{6}$'], fontsize=14)
store_fig(fig, 'brus_dataset_vis_para_data_only')

fig, ax = plt.subplots()
target = u[30,:,-1]
colors = plt.get_cmap('copper')(np.array(range(data_x.shape[0]))/data_x.shape[0])
for i in range(data_x.shape[0]):
    for k in range(data_x.shape[-1]-1):
        data = np.log10(np.abs(data_x[i,:,k]-target))
        ax.scatter(*data, c='blue', s=5)
        if data[1] < -1.5 and data[0] < -2:
            if data[0] < -5:
                ax.text(data[0]+0.18, data[1]-0.16, '$(\mathscr{F}-\mathscr{G})'+f'(U_{{{i}}}^{{{k}}})$', fontsize=14)
            elif data[0] < -3.2:
                ax.text(data[0]-1.1, data[1]-0.5, '$(\mathscr{F}-\mathscr{G})'+f'(U_{{{i}}}^{{{k}}})$', fontsize=14)
            elif data[1]<-2:
                ax.text(data[0]+0.18, data[1]-0.16, '$(\mathscr{F}-\mathscr{G})'+f'(U_{{{i}}}^{{{k}}})$', fontsize=14)
            else:
                ax.text(data[0]-2.4, data[1]-0.16, '$(\mathscr{F}-\mathscr{G})'+f'(U_{{{i}}}^{{{k}}})$', fontsize=14)
tmp = np.log10(np.abs(data_x[:,:,:-1]-target.reshape(1,-1,1)))
temp = np.moveaxis(tmp, 0, -1).reshape(2,-1)
dist = (np.exp(temp)**2).sum(0)
mask = np.logical_not(np.isnan(dist))
sorted_idx = np.argsort(dist[mask])
for i in range(15):
    x, y = temp[:,mask][:,sorted_idx][:,i]
    ax.scatter(x, y, c='red', s=5)

ax.set_xlabel('x coordinate (log)') 
ax.set_ylabel('y coordinate (log)') 
ax.set_title('Plot of $|U - U_{30}^{6}|, U \in D_6$')
fig.tight_layout()
store_fig(fig, 'brus_dataset_vis_para_dist_only')




