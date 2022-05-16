#AUTHOR: ANDREW ENGEL
#andrew.engel@pnnl.gov


import torch

def autograd_components_Fisher(model: torch.nn.Module, y :torch.tensor):
    
    fishers = {}
    
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

        fishers[name] = J_layer.T @ J_layer # An extra transpose operation to my code for us to feel better

        param.requires_grad = False
     
    #reset the model object to be how we started this function
    for i,param in enumerate(model.parameters()):
        if params_that_need_grad[i]:
            param.requires_grad = True #

    return fishers 

def reconstruct_full_fisher_from_components(fisher):
    raise ValueError('Not Implemented Yet')
    #given the dictionary of fisher information matrices, we can 
    #combine them to get the full fisher information matrix.
    return

def autograd_components_Jacobian(model: torch.nn.Module, y :torch.tensor):
    
    Jacobians = {}
    
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

        Jacobians[name] = J_layer # An extra transpose operation to my code for us to feel better

        param.requires_grad = False
     
    #reset the model object to be how we started this function
    for i,param in enumerate(model.parameters()):
        if params_that_need_grad[i]:
            param.requires_grad = True #

    return Jacobians