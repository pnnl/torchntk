#AUTHOR: ANDREW ENGEL
#andrew.engel@pnnl.gov

#NOTES: I adapted get_ntk_n from: Visual Informatics Group @ University of Texas at Austin TENAS project.
#See License for copyright notice of the original author.

#I've also copies some of the utility functions needed directly from pytorch's source code, so I have included their license inside of our license document.

#I've also copied then heavily modified the backward propogation functions in the convolution from the following article by Pierre Jaumier you can read here:
#https://towardsdatascience.com/backpropagation-in-a-convolutional-layer-24c8d64d8509

import numpy as np
import torch
from torch import nn
import copy
from numba import njit

##########--- utility functions---##########

        
@njit
def calc_dw(x,w,b,pad,stride,H_,W_):
    """
    Calculates the derivative of conv(x,w) with respect to w
    
    output is shape:
        [datapoints, in_channels out_filters, kernel_height, kernel_width, out_filters, data_height, data_width
    
    'n f1 f2 c kh kw dh dw -> n (c f1 kh kw) (f2 dh dw)'
    """
    dx, dw, db = None, None, None
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    
    #dw = np.zeros((N,F,F,C,HH,WW,H_,W_),dtype=np.float32)
    
    dw = np.zeros((N,C,F,HH,WW,F,H_,W_),dtype=np.float32)

    xp = zero_pad(x,pad)
    #high priority, how to vectorize this operation?
    for n in range(N):
        for f in range(F):
            for i in range(HH): 
                for j in range(WW): 
                    for k in range(H_): 
                        for l in range(W_): 
                            for c in range(C): 
                                dw[n,c,f,i,j,f,k,l] += xp[n, c, i+stride*k, j+stride*l]                             
    
    return dw.reshape((N,(C*F*HH*WW),(F*H_*W_)))

@njit
def calc_dx(x,w,b,pad,stride,H_,W_):
    '''
    calculates the derivative of conv(x,w) with respect to x
    
    output is a nd-array of shape n x ch_in x og_h x og_w x (h_out w_out ch_out)
    '''
    dx, dw, db = None, None, None
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape 

    dx = np.zeros((C,H,W,F,H_,W_,),dtype=np.float32)
    #high priority, how to vectorize this operation? maybe with np.chunk,split?
    for f in range(F): 
        for i in range(H): 
            for j in range(W):
                for k in range(H_): 
                    for l in range(W_):
                        for c in range(C): 
                            if i-stride*k+pad > HH-1 or j-stride*l+pad > WW-1:
                                continue #this is alternative to padding w with zeros.
                            if i-stride*k+pad < 0 or j-stride*l+pad < 0:
                                continue #this is alternative to padding w with zeros.
                            dx[c,i,j,f,k,l] += w[f, c, i-stride*k+pad, j-stride*l+pad]
    #'c ih iw f oh ow -> (c ih iw) (f oh ow)'
    return dx.reshape(((C*H*W),(F*H_*W_)))
    
@njit
def zero_pad(A,pad):
    N, F, H, W = A.shape
    P = np.zeros((N, F, H+2*pad, W+2*pad),dtype=np.float32)
    P[:,:,pad:H+pad,pad:W+pad] = A
    return P
    
@njit
def cross(X):
    return X.T.dot(X)

def cross_pt_nonp(X,device='cuda'):
    X = X.to(device)
    return X.T.matmul(X)

def cross_pt(X,device='cuda'):
    X = torch.from_numpy(X).to(device)
    X = X.to(device)
    return X.T.matmul(X).cpu()
    
##########--- NTK functions---##########
        


def explicit_ntk(Xs: list, d_int: list, d_array: list, layers: list, d_activationt, device="cuda",) -> list:
    """Calculates the NTK using explicit differentiation for fully connected networks.    
    
    NOTE: all of the input lists should have the same length! 
    NOTE: to get the full ntk, simply sum over the layer dimension of the result: NTK = torch.sum(torch.stack(components),dim=(0))
    
        Parameters: 
            Xs: list, contains all intermediate outputs of each layer in the model
            d_int: list, is the width of each of the dense layers in the model
            d_array: list, has the value of which to sqrt and divide by in each layer
            layers: list, is a list containing the pytorch layers in the model
            d_activationt, a function containing the derivative of the activation function used in all layers but the output layer
            device: str, one of either 'cpu' or 'cuda'; must be the same as the model device location
    
        Returns:
            components: list, of torch.tensor objects, each the ntk 'component' for that layer. 
            Given in backwards order, i.e. starting with the last layer. First weight, then bias.

    """
    components = []
    
    L = len(Xs)-1 #number of layers, Xs goes from inputs to right before outputs; X_0 is the input, X_L CK
    n = Xs[0].shape[-1] #number of datapoints

    #holds the derivatives of activation, first value is empty list...?; just a spacer, replace with array
    Ds_dense = [np.array([[0.0]],dtype=np.float32)] 
    s_matrices = []
    with torch.no_grad():
        ####################################################################################################
        for l in range(0,L):
            if isinstance(layers[l],torch.nn.Linear):
                Ds_dense.append(d_activationt(layers[l](Xs[l].T)).T)
            else:
                Ds_dense.append(np.array([[0.0]],dtype=np.float32))    
        ####################################################################################################
        S = torch.tensor([1.0],dtype=torch.float32).to(device) #this models the backward propogation:   
        for l in range(L,-1,-1):
            if isinstance(layers[l], torch.nn.Linear):
                components.append(cross_pt_nonp(S,device)*cross_pt_nonp(Xs[l],device)/d_array[l])
                if not(hasattr(layers[l], 'bias')) and not(getattr(layers[l], 'bias') is None):
                    W = torch.ones((d_int[l],n),dtype=torch.float32).to(device) * S
                    components.append(cross_pt_nonp(W,device).to(device)/d_array[l])

            #############################
            #now we setup S for the next loop by treating appropriately
            if l==0:
                break

            if isinstance(layers[l], torch.nn.Linear):
                S = torch.matmul(S.T,layers[l].weight).T / torch.sqrt(d_array[l]) #!!!
                if len(S.shape) < 2:
                    S = S[:,None] #expand dimension along axis 1
                if not(isinstance(layers[l-1],float)): #this exludes the reshaping layer
                    S = Ds_dense[l]*S
            
    return components
    
# def explicit_ntk(Ws: list, Ks: list, Xs: list, d_int: list, d_array: list, strides: list, padding: list, layers: list, d_activationt, device="cuda",) -> list:
    # '''    
    # Inputs: 
    # Ws: list, has length number of layers + any reshaping, contains dense layer weight tensors, detatched, and on device
    # Ks: list, has length number of layers + any reshaping, contains 2d convolutional layers weight tensors, detatched, and on device
    # Xs: list, has length number of layers + any reshaping, contains all intermediate outputs of each layer in the models
    # d_int: list, has the number of bias parameters in each dense layer in the models
    # d_array: list, has the value of which to sqrt and divide by in each layer, typically called the NTK normalization. else, its values are 1.
    # strides: list, has the value of stride in each convolutional layer, else 0
    # padding: list, has the value of padding in each convolutional layer, else 0
    # layers: list, is a list containing the pytorch layers in the model, and "0" as a placeholder for a reshaping layer
    # d_activationt, a function containing the derivative of the activation function used in all layers but the output layer, composed of pytorch functions
    # device: str, one of either 'cpu' or 'cuda'; must be the same as the model device location
    
    # NOTE: all of the above lists should have the same length! See example
    
    # OUTPUTS: list of torch.tensor objects, each the ntk 'component' for that layer. Given in backwards order, i.e. starting with the last layer. First weight, then bias.
    
    # NOTE: to get the full ntk, simply sum over the layer dimension of the result: NTK = torch.sum(torch.stack(components),dim=(0))
    
    # updated
    # '''
    # components = []
    
    # L = len(Xs)-1 #number of layers, Xs goes from inputs to right before outputs; X_0 is the input, X_L CK
    # n = Xs[0].shape[-1] #number of datapoints

    #holds the derivatives of activation, first value is empty list...?; just a spacer, replace with array
    # Ds_dense = [np.array([[0.0]],dtype=np.float32)] 
    # Ds_conv = [np.array([[0.0]],dtype=np.float32)]
    # s_matrices = []
    # with torch.no_grad():
        ###################################################################################################
        # for l in range(0,L):
            # if isinstance(layers[l],torch.nn.Linear):
                # Ds_dense.append(d_activationt(layers[l](Xs[l].T)).T)
            # else:
                # Ds_dense.append(np.array([[0.0]],dtype=np.float32))
        ###############################################################################################
        # for l in range(0,L):
            # if isinstance(layers[l],torch.nn.Conv2d):
                # Ds_conv.append(d_activationt(layers[l](Xs[l].T)).reshape(n,-1).T)
            # else:
                # Ds_conv.append(np.array([[0.0]],dtype=np.float32))      
        ###################################################################################################
        # S = torch.tensor([1.0],dtype=torch.float32).to(device) #this models the backward propogation:   
        # for l in range(L,-1,-1):
            # if isinstance(layers[l], torch.nn.Linear):
                # components.append(cross_pt_nonp(S,device)*cross_pt_nonp(Xs[l],device)/d_array[l])
                # if not(hasattr(layers[l], 'bias')) and not(getattr(layers[l], 'bias') is None):
                    # W = torch.ones((d_int[l],n),dtype=torch.float32).to(device) * S
                    # components.append(cross_pt_nonp(W,device).to(device)/d_array[l])

            # elif isinstance(layers[l], torch.nn.Conv2d):
                # if len(S.shape) == 2: #this should only affect the very last layer, at which point, who cares.
                    # S = S[None,:,:]
                    
                # dw = calc_dw(x=Xs[l].T.cpu().numpy(),w=Ks[l].cpu().numpy(),b=0,pad=padding[l],stride=strides[l],H_=Xs[l+1].shape[1],W_=Xs[l+1].shape[0])
                
                # W = torch.matmul(torch.from_numpy(dw).to(device),S.to(device))
                
                # W = torch.diagonal(W,0,0,2)

                # components.append(cross_pt_nonp(W,device).to(device)/d_array[l])
                
                # if not(hasattr(layers[l], 'bias')) and not(getattr(layers[l], 'bias') is None):
                    # N = Ks[l].shape[0]
                    # W = np.split(S.cpu().numpy(),N,axis=1)
                    
                    # W = np.array(W)
                    
                    # W = np.sum(W,axis=(1,2))
                    
                    # components.append(torch.from_numpy(cross(W,)).to(device)/d_array[l])

            ############################
            #now we setup S for the next loop by treating appropriately
            # if l==0:
                # break

            # if isinstance(layers[l], torch.nn.Linear):
                # S = torch.matmul(S.T,Ws[l]).T / torch.sqrt(d_array[l])
                # if len(S.shape) < 2:
                    # S = S[:,None] #expand dimension along axis 1
                # if not(isinstance(layers[l-1],float)): #this exludes the reshaping layer
                    # S = Ds_dense[l]*S
                # else: #and when the reshaping layer occurs we need to apply this instead
                    # S = Ds_conv[l-1]*S

            # elif isinstance(layers[l], torch.nn.Conv2d):
                # dx = calc_dx(x=Xs[l].T.cpu().numpy(),w=Ks[l].cpu().numpy(),b=0,pad=padding[l],stride=strides[l],H_=Xs[l+1].shape[1],W_=Xs[l+1].shape[0])
                # S = (torch.from_numpy(dx[None,:,:]).to(device) @ S) / torch.sqrt(d_array[l])
                # S = Ds_conv[l]*S
            
    # return components



        
