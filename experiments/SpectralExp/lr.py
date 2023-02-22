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

def linear(X):
    return X

def linear_grad(X):
    return X*0 + 1

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

"""**Training Data**"""

SNR = 35
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

gamma1 = 0.1 # gamma1 = d/n
gamma2 = 0.8 # gamma2 = h/n

# L2 regularization
lda = 0
# Learning rate
eta_max = 0.1
eta_min = 0.0001
# Step size
n_max = 4500
n_min = 700
# Number of steps
error = 1e-4

# --------------------------------------------
runs = 1
inc= 300
n_list = np.arange(n_min,n_max,inc)
lr_list = np.array([0.0001,0.001,0.01,0.1])
NN_loss = np.zeros([len(n_list),len(lr_list)])
NN_weight_diff = np.zeros([len(n_list),len(lr_list),3])
for i in tqdm(range(len(n_list))):
    n = n_list[i]
    print(n)
    d = int(n*gamma1)
    h = int(n*gamma2)
    # training and test data with noise
    X,Y,X_test,Y_test = multi_index(n,d,int(2*n/3),sigma=0,ind=1,nonlin=sigmoid)
    for j in tqdm(range(len(lr_list))):
          W = torch.FloatTensor(d,h).normal_(mean=0,std=(1/d)**alpha1).to(device)
          b = torch.FloatTensor(h,1).normal_(mean=0,std=(1/h)**alpha2).to(device)
          #s0= torch.linalg.eigvalsh(W@torch.t(W))
          Ws = W.clone()
          bs = b.clone()
          XX = X@torch.t(X) / d
          CK0 = act(X@W)@torch.t(act(X@W)) / h
          eta_w = lr_list[j]
          eta_b = lr_list[j]
          XW_grad0 = grad(X@W)
          S0 = XW_grad0*((torch.ones(n, 1).to(device))@torch.t(b))
          RF0 = S0@torch.t(S0)
          NTK0 = XX * RF0
          if train_2nd:
              NTK0 = NTK0 + CK0
          ys = act(X@Ws)@bs
          step = 0
          while loss_fn(ys,Y)>error:
              step = step+1
              # update parameters
              dW = torch.t(X) @ (loss_grad(Y, ys) @ torch.t(bs) * grad(X@Ws)) / n
              Ws = Ws + eta_w * (dW - lda*Ws)
              if train_2nd:
                  db = torch.t(X@Ws)@loss_grad(Y, ys) / n
                  bs = bs + eta_b * (db - lda*bs)
              ys = act(X@Ws)@bs
          print("Number of steps: %d"% step)
          XW = act(X@Ws)
          y_NN = act(X_test@Ws)@bs
          NN_loss[i,j] = loss_fn(y_NN,Y_test)
          #s1= torch.linalg.eigvalsh(Ws@torch.t(Ws))
          #NN_weight_diff[i,j,0] = torch.abs(s1[0]-s0[0])
          NN_weight_diff[i,j,0] = torch.linalg.norm(Ws-W, ord=2)
          # CK model
          CK1 = act(X@Ws)@torch.t(act(X@Ws)) / h
          NN_weight_diff[i,j,1] = torch.linalg.norm(CK1-CK0, ord=2)
          # NTK model
          XW_grad = grad(X@Ws)
          S = XW_grad*((torch.ones(n, 1).to(device))@torch.t(bs))
          RF = S@torch.t(S)
          NTKs = XX * RF
          if train_2nd:
              NTKs = NTKs + CK1
          NN_weight_diff[i,j,2] = torch.linalg.norm(NTKs-NTK0, ord=2)
          print("-------------------------------------------------")
          print("dimesion: %d; Training Loss: %f" % (n,loss_fn(ys,Y)))

plt.figure(0)
FONT_SIZE = 10
bin = 50
ylim = [0,100]
plt.rc('font',size=FONT_SIZE)
fig, (ax1,ax2,ax3,ax4) = plt.subplots(1,4,figsize=(22,5))

for j in range(len(lr_list)):
    ax1.plot(n_list,NN_loss[:,j],linewidth=1.2,label='lr=%g'% lr_list[j])
  
    ax2.plot(n_list,NN_weight_diff[:,j,0],linewidth=1.2,label='$\|W_0-W_s\|_2$ lr=%g'% lr_list[j])
    ax3.plot(n_list,NN_weight_diff[:,j,1],linewidth=1.2,label='$\|CK_0-CK_s\|_2$ lr=%g'% lr_list[j])
    ax4.plot(n_list,NN_weight_diff[:,j,2],linewidth=1.2,label='$\|NTK_0-NTK_s\|_2$ lr=%g'% lr_list[j])

    ax1.set_xlabel('Sample Dimension $n$')
    ax1.title.set_text('Test loss, $d/n=$%g, $h/n=$%g'% (gamma1,gamma2))
    ax1.legend()
    ax2.set_xlabel('Sample Dimension $n$')
    ax2.title.set_text('Weight, $d/n=$%g, $h/n=$%g'% (gamma1,gamma2))
    ax2.legend()

    ax3.set_xlabel('Sample Dimension $n$')
    ax3.title.set_text('CK, $d/n=$%g, $h/n=$%g'% (gamma1,gamma2))
    ax3.legend()
    ax4.set_xlabel('Sample Dimension $n$')
    ax4.title.set_text('NTK, $d/n=$%g, $h/n=$%g'% (gamma1,gamma2))
    ax4.legend()

fig.tight_layout(pad=1.5)
fig.savefig('different_lr.png')


