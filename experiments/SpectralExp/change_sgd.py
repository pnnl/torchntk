# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import numpy.random as npr

from tqdm import tqdm
 
import matplotlib 
matplotlib.use('pdf')
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')


import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--steps', type=int, default=9600)
parser.add_argument('--batch', type=int, default=32)
parser.add_argument('--n_max', type=int, default=8000)
parser.add_argument('--n_min', type=int, default=200)
parser.add_argument('--eta', type=float, default=0.1)
parser.add_argument('--gamma1', type=float, default=0.1)
parser.add_argument('--gamma2', type=float, default=1.0)
args = parser.parse_args()

gamma1 = args.gamma1#gamma1 = d/n
gamma2 = args.gamma2 # gamma2 = h/n

# L2 regularization
#lda = args.lda
# Learning rate
eta = args.eta
# Max step size
n_max = args.n_max
n_min = args.n_min
# Number of steps
steps = args.steps
batch = args.batch

def lin(X): 
    return X

# relu
# b0 = 1/4
# b1 = 1/4
def relu(X,normalize=True):
    X[X<0] = 0
    if normalize:
        return np.sqrt(2*np.pi/(np.pi-1))*(X-1/np.sqrt(2*np.pi))
    else:
        return X

def relu_grad(X,normalize=True):
    X[X<0] = 0
    X[X>0] = 1
    if normalize:
        X[X>0] = np.sqrt(2*np.pi/(np.pi-1))
    return X

# softplus
# b0 = 1/4
# b1 = 0.043379
SP = torch.nn.Softplus()
def softplus(X,normalize=True):
    if normalize:
        c0 = 0.806059
        c1 = 0.271514
        return (SP(X)-c0) / (c1)**0.5
    else:
        return SP(X)

def softplus_grad(X,normalize=True):
    if normalize:
        c1 = 0.271514
        return torch.sigmoid(X) / (c1)**0.5
    else:
        return torch.sigmoid(X)

# sigmoid
# b0=0.042692
# b1 =0.002144 
def sigmoid(X,normalize=True):
    X = torch.sigmoid(X)
    if normalize:
        c = np.sqrt(0.21747 / (2 * np.sqrt(2 * np.pi)))
        return (X-0.5)/c
    else:
        return X

def sigmoid_grad(X,normalize=True):
    X = torch.sigmoid(X)
    if normalize:
        c = np.sqrt(0.21747 / (2 * np.sqrt(2 * np.pi)))
        return X * (1-X) / c
    else:
        return X * (1-X)

def tanh(X,normalize=False):
    X = torch.tanh(X)
    if normalize:
        c = 0.627928
        return X / c
    else:
        c = 5
        return X * c

def tanh_grad(X,normalize=False):
    X = torch.tanh(X)
    if normalize:
        c = 0.627928
        return (1 - X**2) / c
    else:
        c = 5
        return (1 - X**2) * c

def quadratic(X,normalize=True):
    if normalize:
        return (X**2-1) / np.sqrt(2)
    else:
        return X**2 / 2

def quadratic_grad(X,normalize=True):
    if normalize:
        return np.sqrt(2)*X
    else:
        return X

# activation function
act = relu
grad = relu_grad

# scaling of parameters
alpha1 = 0.5
alpha2 = 0.5

# whether both layers are optimized
train_2nd = True


def unit_Gaussian(n,d):
    return torch.FloatTensor(n,d).normal_(mean=0,std=1).to(device)  

def single_index(n,d,n1,sigma=0.1,nonlin=relu):
    X = unit_Gaussian(n,d)
    X_test = unit_Gaussian(n1,d)
    theta = torch.FloatTensor(d,1).normal_(mean=0,std=1).to(device)  
    theta = theta / torch.norm(theta)
    if sigma == 0:
        return X, nonlin(X@theta), X_test, nonlin(X_test@theta)
    else:
        return X, nonlin(X@theta) + torch.FloatTensor(n,1).normal_(mean=0,std=sigma).to(device), X_test, nonlin(X_test@theta)

def multi_index(n,d,n1,sigma=0.1,ind=1,nonlin=relu):
    X = unit_Gaussian(n,d)
    X_test = unit_Gaussian(n1,d)
    theta = torch.FloatTensor(d,ind).normal_(mean=0,std=1).to(device) 
    theta = theta / torch.norm(theta)
    a = torch.ones([ind,1]).to(device) / (ind**0.5)
    Y = nonlin(X@theta)@a
    Y_test = nonlin(X_test@theta)@a
    if sigma == 0:
        return X, Y, X_test, Y_test
    else:
        return X, Y + torch.FloatTensor(n,1).normal_(mean=0,std=sigma).to(device), X_test, Y_test


SNR = 5
# std of Gaussian
sigma = (1/SNR)**0.5
# sparsity of target function
spar = 1

print("SNR: %f" % (SNR))
print("noise: %f" % (sigma))


# Y1 - true labels
# Y2 - model prediction

def mse(Y1,Y2):
    return torch.mean((Y1-Y2)**2)

def logistic(Y1,Y2):
    return torch.mean(SP(-Y1*Y2))

def mse_grad(Y1,Y2):
    return Y1 - Y2

def logistic_grad(Y1,Y2):
    return Y1 * torch.sigmoid(-Y1 * Y2)

# ------------------------------------
# Choose objective

loss_fn = mse
loss_grad = mse_grad

"""**Training Neural Networks**"""

# --------------------------------------------
runs = 15
inc= 800
lda = 0
n_list = np.arange(n_min,n_max,inc)
 
NN_loss = np.zeros([len(n_list),runs,2])
NN_weight_diff = np.zeros([len(n_list),runs,8])

for i in tqdm(range(len(n_list))):
    n = n_list[i]
    print(n)
    d = int(n*gamma1)
    h = int(n*gamma2)
    # training and test data with noise
    X,Y,X_test,Y_test = multi_index(n,d,int(2*n/3),sigma=sigma,ind=1,nonlin=sigmoid)
    for j in tqdm(range(runs)):
          W = torch.FloatTensor(d,h).normal_(mean=0,std=(1/d)**alpha1).to(device)
          b = torch.FloatTensor(h,1).normal_(mean=0,std=(1/h)**alpha2).to(device)
          s0 = torch.linalg.eigvalsh(W@torch.t(W))
          Ws = W.clone()
          bs = b.clone()
          XX = X@torch.t(X) / d
          CK0 = act(X@W)@torch.t(act(X@W)) / h
          eta_w = eta
          eta_b = eta
          XW_grad0 = grad(X@W)
          S0 = XW_grad0*((torch.ones(n, 1).to(device))@torch.t(b))
          RF0 = S0@torch.t(S0)
          NTK0 = XX * RF0
          if train_2nd:
              NTK0 = NTK0 + CK0
          s_ntk0 = torch.linalg.norm(NTK0, ord=2)
          for k in range(steps):
              list_sample = torch.randperm(n)
              ys = act(X[list_sample[:batch],:]@Ws)@bs
              # update parameters
              dW = torch.t(X[list_sample[:batch],:]) @ (loss_grad(Y[list_sample[:batch]], ys) @ torch.t(bs) * grad(X[list_sample[:batch],:]@Ws)) / n
              Ws = Ws + eta_w * (dW - lda*Ws)

              if train_2nd:
                  db = torch.t(X[list_sample[:batch],:]@Ws)@loss_grad(Y[list_sample[:batch]], ys) / n
                  bs = bs + eta_b * (db - lda*bs)
          XW = act(X@Ws)
          y_NN = act(X_test@Ws)@bs
          ys = act(X@Ws)@bs
          NN_loss[i,j,0] = loss_fn(ys,Y)
          NN_loss[i,j,1] = loss_fn(y_NN,Y_test)
          s1 = torch.linalg.eigvalsh(Ws@torch.t(Ws))
          NN_weight_diff[i,j,0] = torch.abs(s1[0]-s0[0])
          NN_weight_diff[i,j,1] = torch.linalg.norm(Ws-W, ord=2)
          NN_weight_diff[i,j,2] = torch.linalg.norm(Ws-W, ord='fro')
          # CK model
          CK1 = act(X@Ws)@torch.t(act(X@Ws)) / h
          NN_weight_diff[i,j,3] = torch.linalg.norm(CK1-CK0, ord=2)
          NN_weight_diff[i,j,4] = torch.linalg.norm(CK1-CK0, ord='fro')
          # NTK model
          XW_grad = grad(X@Ws)
          S = XW_grad*((torch.ones(n, 1).to(device))@torch.t(bs))
          RF = S@torch.t(S)
          NTKs = XX * RF
          if train_2nd:
              NTKs = NTKs + CK1
          s_ntk = torch.linalg.norm(NTKs, ord=2)
          NN_weight_diff[i,j,5] = torch.linalg.norm(NTKs-NTK0, ord=2)
          NN_weight_diff[i,j,6] = torch.linalg.norm(NTKs-NTK0, ord='fro')
          NN_weight_diff[i,j,7] = s_ntk-s_ntk0

          print("-------------------------------------------------")
          print("dimesion: %d; Training Loss: %f" % (n,loss_fn(ys,Y)))

plt.figure(0)
FONT_SIZE = 10
plt.rc('font',size=FONT_SIZE)
fig, (ax1,ax2,ax3) = plt.subplots(1,3,figsize=(18,5))
bin = 55

ax1.errorbar(n_list, np.mean(NN_loss[:,:,0],axis=1), yerr=np.std(NN_loss[:,:,0],axis=1),elinewidth=3,label='NN_train')
ax1.errorbar(n_list, np.mean(NN_loss[:,:,1],axis=1), yerr=np.std(NN_loss[:,:,0],axis=1),elinewidth=3,label='NN_test')
#ax1.plot(n_list,NN_loss[:,0],color='g',linewidth=3,label='NN_train')
#ax1.plot(n_list,NN_loss[:,1],color='r',linewidth=3,label='NN_test')
ax1.set_xlabel('Sample Dimension n')
ax1.set_ylabel('Final training and test Losses')
ax1.title.set_text('Losses $d/n=$%g'% gamma1)
ax1.legend()

#histeig(s0.cpu().numpy(), ax2, xgrid=None, dgrid=None, bins=bin, xlim=None, ylim=None, title='initial')
#histeig(s1.cpu().numpy(), ax2, xgrid=None, dgrid=None, bins=bin, xlim=None, ylim=None, title=None)
ax2.hist(s0.cpu().numpy(),histtype='stepfilled', alpha=0.2, bins=bin,label='Initialized')
ax2.hist(s1.cpu().numpy(),histtype='stepfilled', alpha=0.2, bins=bin,label='Trained')
n_fin = n_max-inc
ax2.set_xlabel('Eigenvalues of $WW^T$ when n=%d'% n_fin)
ax2.title.set_text('Relu activation, lr=%g'% eta)
ax2.legend()

ax3.errorbar(n_list, np.mean(NN_weight_diff[:,:,0],axis=1), yerr=np.std(NN_weight_diff[:,:,0],axis=1),elinewidth=3,label='Largest eigenvalue difference')
ax3.errorbar(n_list, np.mean(NN_weight_diff[:,:,1],axis=1), yerr=np.std(NN_weight_diff[:,:,1],axis=1),elinewidth=3,label='$\|W_0-W_s\|_2$')
ax3.errorbar(n_list, np.mean(NN_weight_diff[:,:,2],axis=1), yerr=np.std(NN_weight_diff[:,:,2],axis=1),elinewidth=3,label='$\|W_0-W_s\|_{Fro}$')
#ax3.plot(n_list,NN_weight_diff[:,0],color='g',linewidth=3,label='Largest eigenvalue difference')
#ax3.plot(n_list,NN_weight_diff[:,1],color='r',linewidth=3,label='$\|W_0-W_s\|_2$')
ax3.set_xlabel('Sample Dimension n')
ax3.title.set_text('Weight differences, $h/n=$%g'% gamma2)
ax3.legend()

fig.tight_layout(pad=1.5)
fig.savefig('sgd_relu_weight_%g_%g_lr=%g_b=%d.png'% (gamma1,gamma2,eta,batch))

# spectrum of NTK and CK
s2 = torch.linalg.eigvalsh(CK0)
s3 = torch.linalg.eigvalsh(CK1)
s4 = torch.linalg.eigvalsh(NTK0)
s5 = torch.linalg.eigvalsh(NTKs)

plt.figure(0)
FONT_SIZE = 12
plt.rc('font',size=FONT_SIZE)
fig, (ax1,ax2,ax3,ax4) = plt.subplots(1,4,figsize=(22,5))
bin = 50

#histeig(s2.cpu().numpy(), ax2, xgrid=None, dgrid=None, bins=bin, xlim=None, ylim=None, title='initial')
#histeig(s3.cpu().numpy(), ax2, xgrid=None, dgrid=None, bins=bin, xlim=None, ylim=None, title=None)
ax1.hist(s2.cpu().numpy(),histtype='stepfilled', alpha=0.2, bins=bin,label='Initialized')
ax1.hist(s3.cpu().numpy(),histtype='stepfilled', alpha=0.2, bins=bin,label='Trained')
n_fin = n_max-inc
ax1.set_ylim([0,150])
ax1.set_xlabel('Eigenvalues of CK when n=%d'% n_fin)
ax1.title.set_text('SGD,  Batch=%d'% batch)
ax1.legend()

#histeig(s4.cpu().numpy(), ax2, xgrid=None, dgrid=None, bins=bin, xlim=None, ylim=None, title='initial')
#histeig(s5.cpu().numpy(), ax2, xgrid=None, dgrid=None, bins=bin, xlim=None, ylim=None, title=None)
ax2.hist(s4.cpu().numpy(),histtype='stepfilled', alpha=0.2, bins=bin,label='Initialized')
ax2.hist(s5.cpu().numpy(),histtype='stepfilled', alpha=0.2, bins=bin,label='Trained')
n_fin = n_max-inc
ax2.set_ylim([0,150])
ax2.set_xlabel('Eigenvalues of NTK when n=%d'% n_fin)
ax2.title.set_text('Relu activation, lr=%g'% eta)
ax2.legend()

ax3.errorbar(n_list, np.mean(NN_weight_diff[:,:,3],axis=1), yerr=np.std(NN_weight_diff[:,:,3],axis=1),elinewidth=3,label='$\|CK_0-CK_s\|_2$')
ax3.errorbar(n_list, np.mean(NN_weight_diff[:,:,4],axis=1), yerr=np.std(NN_weight_diff[:,:,4],axis=1),elinewidth=3,label='$\|CK_0-CK_s\|_{Fro}$')
ax3.set_xlabel('Sample Dimension $n$')
ax3.title.set_text('$d/n=$%g, $h/n=$%g'% (gamma1,gamma2))
ax3.legend()


ax4.errorbar(n_list, np.mean(NN_weight_diff[:,:,5],axis=1), yerr=np.std(NN_weight_diff[:,:,5],axis=1),elinewidth=3,label='$\|NTK_0-NTK_s\|_2$')
ax4.errorbar(n_list, np.mean(NN_weight_diff[:,:,6],axis=1), yerr=np.std(NN_weight_diff[:,:,6],axis=1),elinewidth=3,label='$\|NTK_0-NTK_s\|_{Fro}$')
ax4.errorbar(n_list, np.mean(NN_weight_diff[:,:,7],axis=1), yerr=np.std(NN_weight_diff[:,:,7],axis=1),elinewidth=3,label='$\lambda_\max(NTK_s)-\lambda_\max(NTK_0)$')
ax4.set_xlabel('Sample Dimension $n$')
ax4.title.set_text('$d/n=$%g, $h/n=$%g'% (gamma1,gamma2))
ax4.legend()

fig.tight_layout(pad=1.5)
fig.savefig('relu_ck_ntk_%g_%g_lr=%g_b=%d.png'% (gamma1,gamma2,eta,batch))

np.save('relu_NN_history_%g_%g_lr=%g_b=%d.npy' % (gamma1,gamma2,eta,batch), NN_loss)
np.save('relu_spectra_history_%g_%g_lr=%g_b=%d.npy' % (gamma1,gamma2,eta,batch), NN_weight_diff)

n = n_max
print(n)
d = int(n*gamma1)
h = int(n*gamma2)
# training and test data with noise
X,Y,X_test,Y_test = multi_index(n,d,int(2*n/3),sigma=sigma,ind=1,nonlin=sigmoid)
W = torch.FloatTensor(d,h).normal_(mean=0,std=(1/d)**alpha1).to(device)
b = torch.FloatTensor(h,1).normal_(mean=0,std=(1/h)**alpha2).to(device)
Ws = W.clone()
bs = b.clone()
history = np.zeros([steps,2])
for k in range(steps):
    list_sample = torch.randperm(n)
    ys = act(X[list_sample[:batch],:]@Ws)@bs
    # update parameters
    dW = torch.t(X[list_sample[:batch],:]) @ (loss_grad(Y[list_sample[:batch]], ys) @ torch.t(bs) * grad(X[list_sample[:batch],:]@Ws)) / n
    Ws = Ws + eta_w * (dW - lda*Ws)
    if train_2nd:
        db = torch.t(X[list_sample[:batch],:]@Ws)@loss_grad(Y[list_sample[:batch]], ys) / n
        bs = bs + eta_b * (db - lda*bs)
    XW = act(X@Ws)
    y_NN = act(X_test@Ws)@bs
    ys = act(X@Ws)@bs
    history[k,0] = loss_fn(ys,Y)
    history[k,1] = loss_fn(y_NN,Y_test)
print("-------------------------------------------------")
print("dimesion: %d; Training Loss: %f" % (n,loss_fn(ys,Y)))
plt.figure(0)
FONT_SIZE = 12
plt.rc('font',size=FONT_SIZE)
fig, ax = plt.subplots(1,1)
step_list = np.arange(0,steps)
ax.plot(step_list,history[:,0],color='g',linewidth=3,label='NN_train')
ax.plot(step_list,history[:,1],color='y',linewidth=3,label='NN_test')
#ax1.plot(n_list,NN_loss[:,1],color='r',linewidth=3,label='NN_test')
ax.set_xlabel('Epochs')
ax.set_ylabel('Training and Test Losses with SGD')
ax.title.set_text('Losses $d/n=$%g, $h/n=$%g,lr=%g,b=%d'% (gamma1,gamma2,eta,batch))
ax.legend()

fig.tight_layout(pad=1.5)
fig.savefig('Train_%g_%g_lr=%g_n=%d_b=%d.png'% (gamma1,gamma2,eta,n_max,batch))
