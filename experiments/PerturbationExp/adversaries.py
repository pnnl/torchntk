#This file implements a variety of gradient-based adversarial attacks on neural net pytorch models

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim





#The FGSM attack

#assumes that either image.retain_grad() or image.requires_grad=True were called before this function call
#input_image: the point at which the model will be attacked
#label: the true label of the image
#epsilon: the attack radius
#model: the neural net under attack, assumed to extend nn.Module
#loss_function: the loss function to minimize/maximize for finding the adversarial perturbation
#p: the attack is in the ell_p norm
#clip_min, clip_max (optional): pixel values for images typically fall in a certain range. If both clip_min,clip_max are set to floats, then the adversarial image is threshholded at clip_min and clip_max. If both are None, then the image is not threshholded. If of clip_min,clip_max are none and the other is not, then the method throws an error
#targeted (optional): True for a targeted attack, false otherwise
#output_perturbation (optional): if True, the method returns the adversarial perturbation. If false, the method returns the image plus the perturbation.
#device: the device on which to compute the attack
def fgsm_attack_p(input_image, label, epsilon,model, loss_function,p,clip_min=None,clip_max=None, targeted=False, output_perturbation=False,device=torch.device('cpu')):
    #pre_processing
    image=input_image.detach()

    image.requires_grad=True
    image=image+torch.zeros_like(image)
    image.retain_grad()
    label=torch.tensor([label])
    label=label.to(device)
    #calculate the gradient
    model.zero_grad()
    pred=model(image)
    loss=loss_function(pred,label)
    loss.backward()
      #data_grad=image.grad.data
    data_grad=image.grad.data
    if clip_min is None and clip_max is not None or clip_max is None and clip_min is not None:
        raise ValueError("One of clip_min and clip_max is None, then need to be either both None or both not None")
    if targeted: #for a targeted attack we want to minimize the risks of the target rather than maximize the risk of the current point
        data_grad=-data_grad
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    if torch.all(torch.eq(data_grad,torch.zeros_like(data_grad))):
        perturbation=torch.zeros_like(data_grad)
    elif p==1:
        mx=data_grad.max()
       #perturbation=torch.zeros_like(data_grad)
        perturbation= (data_grad== mx)*sign_data_grad


        nrm=torch.dist(data_grad,torch.zeros_like(data_grad), 1)    
        perturbation=perturbation/nrm
    elif p==float('inf'): 
        perturbation=sign_data_grad;
    else:
        q=1/(1-1/p)
        perturbation=  data_grad.abs().pow(q-1).mul(sign_data_grad)
        nrm=torch.dist(data_grad,torch.zeros_like(data_grad), q);
        perturbation=perturbation/nrm.pow(q-1)
     # Create the perturbed image by adjusting each pixel of the input image
    perturbation=epsilon*perturbation
    perturbed_image = image + perturbation
    # Adding clipping to maintain [0,1] range
    if clip_min is not None:
        perturbed_image = torch.clamp(perturbed_image, clip_min, clip_max)
        perturbation=perturbed_image-image
    # Return the perturbed image
    if output_perturbation:
        return perturbation.detach()
    else:
        return perturbed_image.detach()





#The PGD attack

#input_image: the point at which the model will be attacked
#label: the true label of the image
#epsilon: the attack radius
#model: the neural net under attack, assumed to extend nn.Module
#loss_function: the loss function to minimize/maximize for finding the adversarial perturbation
#p: the attack is in the ell_p norm. Only p=2 and p=float('inf') are implement. for other values of p, this method throws an error
#iterations: the numper of steps in the PGD attack
#step_size:(optional) the step size of the PGD attack. If step_size=None, then the step size is chosen to be 2*epsilon/iterations*1.4. Reasoning: 2*epsilon is the diameter of the ball, so we want the step size to be large enough to reach any place in the ball when starting at a random initialization, and a little larger for projecting back down to the ball a few times.
#rand_init:(optional) If false, the optimization starts from 'image'. Otherwise, the optimization starts at a random point in the epsilon ball around image according to the uniform distribution
#clip_min, clip_max:(optional) pixel values for images typically fall in a certain range. If both clip_min,clip_max are set to floats, then the adversarial image is threshholded at clip_min and clip_max. If both are None, then the image is not threshholded. If of clip_min,clip_max are none and the other is not, then the method throws an error
#intermediate_clip: (optional) whether thresholding is applied after each step of PGD or just at the end. Notice there are only convergence guarantees for p=inf
#targeted:(optional) True for a targeted attack, false otherwise
#output_perturbation:(optional) if True, the method returns the adversarial perturbation. If false, the method returns the image plus the perturbation.
#device: the device on which to compute the attack
def pgd_attack_p(input_image,label, epsilon,model, loss_function,p,iterations,step_size=None, rand_init=False, clip_min=None,clip_max=None, intermediate_clip=False, targeted=False,output_perturbation=False,device=torch.device('cpu')):
    image=input_image.detach()
    image.requires_grad=True
    if intermediate_clip is True and (clip_min is None or clip_max is None):
        raise ValueError("if intermediate_clip is True, both clip_min and clip_max should not be None")
    if p!=float('inf') and p!=2:
        raise ValueError("p must be either 2 or infinity, other p-norms are not implemented")
    perturbation_step=torch.zeros_like(image)
    if rand_init:
        if p==float('inf'):
            perturbation=epsilon*(2*torch.rand_like(image, dtype=torch.float)-1)
        else:
            perturbation=torch.randn_like(image,dtype=torch.float)
            nrm=torch.dist(perturbation,torch.zeros_like(perturbation), p)
            perturbation=perturbation/nrm
            perturbation=perturbation*epsilon*torch.rand(torch.tensor([1]))

    else:
        perturbation=torch.zeros_like(image)

    #heuristic for choosing a step size: we want the step size to be large enough to cross the epsilon-ball end-to-end and then some for getting lost in high dimensional space
    if step_size is None:
        step_size=2*epsilon/iterations*1.4
    curr_image=image+perturbation
    curr_image.retain_grad()
    for i in range(iterations):
        #take a step in the direction of +/- gradient
        #perturbation_step=fgsm_attack_p(curr_image,label,step_size,model,loss_function,p,targeted=targeted,output_perturbation=True,clip_min=clip_min,clip_max=clip_max)
        #curr_image=curr_image.detach()+perturbation_step.detach()
        curr_image=fgsm_attack_p(curr_image,label,step_size,model,loss_function,p,targeted=targeted,output_perturbation=False,clip_min=clip_min,clip_max=clip_max,device=device)
        if clip_min is not None: #note that one None and the other not none was checked in the fgsm call
            curr_image=torch.clamp(curr_image,clip_min,clip_max)
        perturbation=curr_image-image
        

        #next we check if we are in the norm ball and otherwise project
        nrm=torch.dist(perturbation,torch.zeros_like(perturbation), p);
        if nrm>epsilon:
            if p== float('inf'):
                perturbation=torch.clamp(perturbation, -1*epsilon, epsilon)
            else:
                perturbation=perturbation *(epsilon/nrm);
            curr_image=image.detach()+perturbation.detach()
            if intermediate_clip and clip_min is not None: #note that one None and the other not none was checked in the fgsm call
                curr_image=torch.clamp(curr_image,clip_min,clip_max)
            perturbation=curr_image-image
        
            nrm=torch.dist(perturbation,torch.zeros_like(perturbation), p);
        curr_image.requires_grad=True
    if output_perturbation:
        return perturbation.detach()
    else:
        return curr_image.detach()


#Repetitions of the PGD attack from random initializations

#input_image: the point at which the model will be attacked
#label: the true label of the image
#epsilon: the attack radius
#model: the model under attack, assumed to extend nn.Module
#loss_function: the loss function to minimize/maximize for finding the adversarial perturbation
#p: the attack is in the ell_p norm. Only p=2 and p=float('inf') are implement. for other values of p, this method throws an error
#iterations: the numper of steps in the PGD attack
#repetitions: number of times to repeat the pgd attack
#step_size:(optional) the step size of the PGD attack. If step_size=None, then the step size is chosen to be 2*epsilon/iterations*1.4. Reasoning: 2*epsilon is the diameter of the ball, so we want the step size to be large enough to reach any place in the ball when starting at a random initialization, and a little larger for projecting back down to the ball a few times.
#include_center_initialization(optional): whether or not to include input_image as the initialization point for one of the repetitions
#clip_min, clip_max:(optional) pixel values for images typically fall in a certain range. If both clip_min,clip_max are set to floats, then the adversarial image is threshholded at clip_min and clip_max. If both are None, then the image is not threshholded. If of clip_min,clip_max are none and the other is not, then the method throws an error
#intermediate_clip: (optional) whether thresholding is applied after each step of PGD or just at the end. Notice there are only convergence guarantees for p=inf
#targeted:(optional) True for a targeted attack, false otherwise
#output_perturbation:(optional) if True, the method returns the adversarial perturbation. If false, the method returns the image plus the perturbation.
#device: the device on which to compute the attack
def pgd_repetitions(input_image,label, epsilon,model, loss_function,p,iterations,repetitions=10,step_size=None, include_center_initialization=True, clip_min=None,clip_max=None, intermediate_clip=False, targeted=False,output_perturbation=False,device=torch.device('cpu')):
    example=input_image
    score=loss_function(model(input_image),label)
    if include_center_initialization:
        repetitions-=1
    for i in range(repetitions):
        atk= pgd_attack_p(input_image,label,epsilon,model,loss_function,p,iterations,step_size=step_size,rand_init=True,clip_min=clip_min,clip_max=clip_max,intermediate_clip=intermediate_clip,targeted=targeted, output_perturbation=output_perturbation,device=device)
        atk_score=loss_function(model(atk),label)
        if atk_score>score:
            score=atk_score
            example=atk
    if include_center_initialization:
        atk= pgd_attack_p(input_image,label,epsilon,model,loss_function,p,iterations,rand_init=False)
        atk_score=loss_function(model(atk),label)

        if atk_score>score:
            score=atk_score
            example=atk
    return example 




#returns x
def trivial_attack(model, x,y):
    return x


#For verctorizing an aversary that can attack a single data point

#model: the model under attack
#xs: a tensor of data points. The first index is assumed to index different data points, the other indices index the dimensions of the other data points
#ys: the labels corresponding to each data point
#attack: the unvectorized attack. The function signature is assumed to be attack(model, x,y), where x,y are both torch tensors representing the datapoint and the label respectively and model is a pytorch model assumed to extend nn.Module
def vectorize_adversary(model, xs,ys,attack):
    adversarial_examples_list= [None]*xs.size(0)
    index=0
    for x in xs:
        adversarial_examples_list[index]=attack(model, x,ys[index])
        index+=1
    adversarial_examples=torch.stack(adversarial_examples_list)
    return adversarial_examples








