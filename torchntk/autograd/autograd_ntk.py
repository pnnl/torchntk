#AUTHOR: ANDREW ENGEL
#andrew.engel@pnnl.gov


import numpy as np
import torch
from torch import nn
import copy

# ---UTILITY---
def _del_nested_attr(obj, names):
    """
    Deletes the attribute specified by the given list of names.
    For example, to delete the attribute obj.conv.weight,
    use _del_nested_attr(obj, ['conv', 'weight'])
    """
    if len(names) == 1:
        delattr(obj, names[0])
    else:
        _del_nested_attr(getattr(obj, names[0]), names[1:])

def _set_nested_attr(obj, names, value):
    """
    Set the attribute specified by the given list of names to value.
    For example, to set the attribute obj.conv.weight,
    use _del_nested_attr(obj, ['conv', 'weight'], value)
    """
    if len(names) == 1:
        setattr(obj, names[0], value)
    else:
        _set_nested_attr(getattr(obj, names[0]), names[1:], value)

def extract_weights(mod):
    """
    This function removes all the Parameters from the model and
    return them as a tuple as well as their original attribute names.
    The weights must be re-loaded with `load_weights` before the model
    can be used again.
    Note that this function modifies the model in place and after this
    call, mod.parameters() will be empty.
    """
    orig_params = tuple(mod.parameters())
    # Remove all the parameters in the model
    names = []
    for name, p in list(mod.named_parameters()):
        _del_nested_attr(mod, name.split("."))
        names.append(name)

    # Make params regular Tensors instead of nn.Parameter
    params = tuple(p.detach().requires_grad_() for p in orig_params)
    return params, names

def load_weights(mod, names, params):
    """
    Reload a set of weights so that `mod` can be used again to perform a forward pass.
    Note that the `params` are regular Tensors (that can have history) and so are left
    as Tensors. This means that mod.parameters() will still be empty after this call.
    """
    for name, p in zip(names, params):
        _set_nested_attr(mod, name.split("."), p)


# ---Main---
def naive_ntk(model,x,device='cpu',MODE='samples'):
    """Calculates the NTK for a model, p_dict a state dictionary, and x, a single tensor fed into the model.
    
    The NTK is the grammian of the Jacobian of the model output to w.r.t. the weights of the model
    
    This function will output the NTK such that the minima matrix size is used. If the Jacobian is an NxM
    matrix, then the NTK is formulated so that if N < M; NTK is NxN. If M<N, then NTK is MxM.
    
        parameters:
            model: torch.nn.Module 
            x: torch.Tensor
            device: 'cpu',
            MODE: 'minima'
    
        returns:
            NTK: torch.Tensor
    
    #EXAMPLE USAGE:
    device='cpu'
    model = MODEL() #a torch.nn.Module object 
    model.to(device)

    x_test = np.ones((100,1,28,28),dtype=np.float32)
    x_test = torch.from_numpy(x_test)

    NTK = calculate_NTK(model,x_test)
    
    """
    if not(MODE in ['minima','samples','params']):
        raise ValueError("MODE must be one of 'minima','samples','params'")
    
    x = x.to(device)
    x.requires_grad=False
    N = x.shape[0]
    M = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    #We need to create a clone of the model or else we make it unusable as part of the trickery 
    #to get pytorch to do what we want. Unforutantely, this exlcludes super big models. but, eh.
    model_clone = copy.deepcopy(model)
    
    params, names = extract_weights(model_clone)
    def model_ntk(*args,model=model_clone, names=names):
        params = tuple(args)
        load_weights(model, names, params)
        return model(x)
    
    Js = torch.autograd.functional.jacobian(model_ntk, tuple(params), create_graph=False, vectorize=True)
    
    Js = list(Js) #needs to be mutable, tuples aren't mutable
    #collapse the tensors
    for i,tensor in enumerate(Js):
        Js[i] = tensor.reshape(N,-1)
    
    J = torch.cat(Js,axis=1)
    
    if MODE=='minima':
        if N < M: #if datasize points is less than number of parameters:
            NTK = torch.matmul(J,J.T)

        if N >= M:#if number of parameters is less than datasize:
            NTK = torch.matmul(J.T,J)
    elif MODE=='samples':
        NTK = torch.matmul(J,J.T)
    elif MODE=='params':
        NTK = torch.matmul(J.T,J)
    
    return NTK
   
def autograd_components_ntk(model: torch.nn.Module, y :torch.Tensor):
    """calulcates each layerwise component and returns a dictionary whose keys are the named parameters, e.g. "l1.weight".
    
        parameters:
            model: a torch.nn.Module object. Must terminate to a single neuron output
            y: the final single neurn output of the model evaluated on some data
    
        returns:
            NTKs: a dictionary whose keys are the named parameters and values are torch.Tensors representing those parameters additive contribution to the NTK
    """
    NTKs = {}
    
    if len(y.shape) > 1:
        raise ValueError('y must be 1-D, but its shape is: {}'.format(y.shape))
    
    params_that_need_grad = []
    for param in model.parameters():
        if param.requires_grad:
            params_that_need_grad.append(param.requires_grad)
            #first set all gradients to not calculate, time saver
            param.requires_grad = False
        else:
            params_that_need_grad.append(param.requires_grad)

    #how do we parallelize this operation across multiple gpus or something? that be sweet.
    for i,z in enumerate(model.named_parameters()):
        if not(params_that_need_grad[i]): #if it didnt need a grad, we can skip it.
            continue
        name, param = z
        param.requires_grad = True #we only care about this tensors gradients in the loop
        this_grad=[]
        for i in range(len(y)): #first dimension must be the batch dimension
            model.zero_grad()
            y[i].backward(create_graph=True)
            this_grad.append(param.grad.detach().reshape(-1).clone())

        J_layer = torch.stack(this_grad) # [N x P matrix] #this will go against our notation, but I'm not adding

        NTKs[name] = J_layer @ J_layer.T # An extra transpose operation to my code for us to feel better

        param.requires_grad = False
     
    #reset the model object to be how we started this function
    for i,param in enumerate(model.parameters()):
        if params_that_need_grad[i]:
            param.requires_grad = True #

    return NTKs

def autograd_ntk(model: torch.nn.Module, y :torch.Tensor):
    """calulcates each layerwise component and returns a torch.Tensor representing the NTK
    
        parameters:
            model: a torch.nn.Module object. Must terminate to a single neuron output
            y: the final single neuron output of the model evaluated on some data
    
        returns:
            NTK: a torch.Tensor representing the emprirical neural tangent kernel of the model
    """
    NTK = False
    
    if len(y.shape) > 1:
        raise ValueError('y must be 1-D, but its shape is: {}'.format(y.shape))
    
    params_that_need_grad = []
    for param in model.parameters():
        if param.requires_grad:
            params_that_need_grad.append(param.requires_grad)
            #first set all gradients to not calculate, time saver
            param.requires_grad = False
        else:
            params_that_need_grad.append(param.requires_grad)

    #how do we parallelize this operation across multiple gpus or something? that be sweet.
    for i,z in enumerate(model.named_parameters()):
        if not(params_that_need_grad[i]): #if it didnt need a grad, we can skip it.
            continue
        name, param = z
        param.requires_grad = True #we only care about this tensors gradients in the loop
        this_grad=[]
        for i in range(len(y)): #first dimension must be the batch dimension
            model.zero_grad()
            y[i].backward(create_graph=True)
            this_grad.append(param.grad.detach().reshape(-1).clone())

        J_layer = torch.stack(this_grad) # [N x P matrix] #this will go against our notation, but I'm not adding

        if (type(NTK) is bool) and not(NTK):
            NTK = J_layer @ J_layer.T # An extra transpose operation to my code for us to feel better
        else:
            NTK += J_layer @ J_layer.T

        param.requires_grad = False
     
    #reset the model object to be how we started this function
    for i,param in enumerate(model.parameters()):
        if params_that_need_grad[i]:
            param.requires_grad = True #

    return NTK    
   
#Old autograd_ntk. I expect its worse because it gets things one row at a time.
def old_autograd_ntk(xloader, network, num_batch=-1,device='cpu'):
    """Find the NTK of a network with the supplied by 'xloader'.

    NOTE: Adapted FROM: https://github.com/VITA-Group/TENAS
    
        parameters:
            xloader: torch.data.utils.DataLoader object whose first value are inputs to the model
            network: torch.nn.Module object that terminates in a single neuron output
            num_batch: how many batches from xloader to read to evaluate the NTK. -1 uses all the data
            device: str, either 'cpu' or 'cuda'
    
    """

    ntks = []
    network.eval()

    grads = [[] for _ in range(1)] #num_layers
    for i,(inputs,target)  in enumerate(xloader):
        #this line checks to see if we can exit early;
        if num_batch > 0 and i >= num_batch: break

        if device=='cuda':
            inputs = inputs.cuda(device=device, non_blocking=True)

        net_idx=0
        network.zero_grad()

        if device=='cuda':
            inputs_ = inputs.clone().cuda(device=device, non_blocking=True)
        else:
            inputs_ = inputs.clone()

        logit = network(inputs_)
        
        assert logit.shape[1]==1

        for _idx in range(len(inputs_)):
            gradient = torch.ones_like(logit[_idx:_idx+1])
            logit[_idx:_idx+1].backward(gradient, retain_graph=True)
            grad = []
            for name, W in network.named_parameters():
                if W.grad is not None:
                    grad.append(W.grad.view(-1).detach())
            grads[net_idx].append(torch.cat(grad, -1))
            network.zero_grad()

            if device == 'cuda':
                #Interesting! Tenas also needed to vacate the cache to get their code to run...
                torch.cuda.empty_cache()
                pass
    grads = [torch.stack(_grads, 0) for _grads in grads]
    ntks = [torch.einsum('nc,mc->nm', [_grads, _grads]) for _grads in grads]

    return ntks[0]
    

    
    
