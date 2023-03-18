#fundamental methods for the smallest_adversarial_example experiments

import sys
import numpy as np
from matplotlib import pyplot as plt
import pickle
import numpy.linalg as la
import ntk_methods
import math
from torchvision import datasets, transforms
import torch.nn.functional as F
import os



#given a neural net and an attack method with function signature attack(x,label,epsilon,model), finds the smallest epsilon for which attack(x,label,epsilon,model) is misclassified by the neural net using a binary search
#returns the smallest epsilon for which one can find an adversarial example along with this adversarial example

#x: data point at which we want to find the smallest adversarial perturbation
#label: true label of the data point
#attack: an attack with function signature attack(x,label,epsilon,model) which returns an adversarial example
#upper_bound_guess: an upper bound on the size of the smallest adversarial perturbation 
#strict: if True, upper_bound_guess is treated as a strict upper bound on the size of the smallest adversarial perturbation, otherwise it's assumed to be just a guess (setting this variable as true produces a significant speedup)
#error: stopping condition for the binary search-- the binary search terminates when upper_bound - lower bound <error
#fail_value: this value is returned if an adversarial example is not found
def smallest_adversarial_perturbation(x, label,model, attack, upper_bound_guess=1,strict=False,error=10**-4,fail_value=None):
    pred=model(x)
    _, predicted = pred.max(1)
    if not predicted.eq(label).item():
        adversarial_radius=0
        example=x
    else:
        mx_float=sys.float_info.max
        correct=True#if the upper bound is correct
        under_mx_float=True
        while not strict and under_mx_float and correct:
            if upper_bound_guess>=mx_float:
                under_mx_float=False
            upper_bound_guess=min(upper_bound_guess,mx_float)
            try:
                adversarial_example=attack(x, label, upper_bound_guess,model)
                pred=model(adversarial_example)
                _, predicted = pred.max(1)
                correct=predicted.eq(label).item()
                upper_bound_guess*=2
            except:
                under_mx_float=False
                ub=fail_value
                adversarial_example=[]
                break
        if under_mx_float or strict:
            ub=upper_bound_guess #Upper bound
            lb=0 #lower bound
            while ub-lb>error:
                eps_mid=(ub+lb)/2
                adversarial_example=attack(x, label, eps_mid,model)
                pred=model(adversarial_example)
                _, predicted = pred.max(1)
                correct=predicted.eq(label).item()
                if correct:
                    lb=eps_mid
                else:
                    ub=eps_mid

                
            adversarial_radius=ub
            example=adversarial_example

        else:
            adversarial_radius=fail_value
            example=None

    return adversarial_radius, example



#given a nested list values_list for which values_list[i][j] are scalars, plots a histogram of the aggregate of all the scalars in the nested list, with error bars given by computing the standard deviation of the histograms of values_list[i]


#values_list: A list of numpy arrays
#bins: for specifying the binds of the histogram. If None, determined automatically
#save_file: path at which to save the final plot
#error_bars: whether to plot error bars on the histogram
#x_label: label on x-axis
#fail_value: exclude these instances from the histogram
#rng: range of histogram
#title: title of histogram
def attribute_histogram(values_list,bins=None,save_file=None,error_bars=True,x_label='x',fail_value=None,rng=None,title=None):
    all_values=[v  for values in values_list for v in values if v is not fail_value]
    all_values=np.array(all_values)
    weights=np.ones_like(all_values)*1/len(all_values)
    if bins is None:
        y, binEdges=np.histogram(all_values,weights=weights,range=rng)    
    else:
        y, binEdges=np.histogram(all_values,bins=bins,weights=weights,range=rng)
    hist_vals=[]
    for values in values_list:
        rm_values=[v for v in values if v is not fail_value]
        #hst, edge_vals=np.histogram(rm_values,bins=binEdges,range=rng)
        if bins is None:
            hst, edge_vals=np.histogram(rm_values,range=rng)
        else:
            hst, edge_vals=np.histogram(rm_values,bins=bins,range=rng)
        hist_vals.append(hst)
    hist_vals=np.array(hist_vals)
    stds=np.std(hist_vals,axis=0)/len(all_values)


    bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
    width      = 0.05
    #hist_plot=plt.hist(all_values,bins=bins,weights=weights,range=rng)
    if error_bars:
        plt.bar(bincenters, y, width=width, color='b', yerr=stds,capsize=5.0)
    plt.ylabel('Average Frequency')
    plt.xlabel(x_label)
    if title is not None:
        plt.title(title)
    if save_file is None:
        plt.show()
    else:
        plt.savefig(save_file+".png",bbox_inches='tight')
    plt.close()


#computes the cosine similarity of a single data point with the entire feature mapping given by the NTK

#net: a neural net
#feature mapping: a numpy array that contains as columns the feature mapping given by the ntk
#x: a data point
def cosine_similarity(net,feature_mapping,x):
    ftr= ntk_methods.feature_mapping(net ,x,data_set=False,return_torch_tensor=False)
    tpl=la.lstsq(feature_mapping,ftr)
    proj_coeffs=tpl[0]
    x_proj=np.dot(feature_mapping,proj_coeffs)
    cos_sim=np.dot(np.transpose(x_proj),ftr)/(la.norm(ftr)*la.norm(x_proj))
    return cos_sim
    #return math.acos(la.norm(x_proj)/la.norm(ftr))


#computes the cosine similarity of multiple data points with the entire feature mapping given by the NTK. Vectorization should produce a significant speedup
#test_features: mapping of the test data under the feature mapping given by the NTK
#train_features: mapping of the training data under the feature mapping given by the NTK
def cosine_similarity_vectorized(test_features,train_features):
    tpl=la.lstsq(train_features,test_features)
    proj_coeffs=tpl[0]
    x_projs=np.dot(train_features,proj_coeffs)
    num_points=x_projs.shape[1]
    cos_sims=[-2]*num_points
    for i in range(num_points):
        x_proj=x_projs[:,i]
        ftr=test_features[:,i]
        cos_sims[i]=np.dot(np.transpose(x_proj),ftr)/(la.norm(ftr)*la.norm(x_proj))
    return cos_sims
    

#computes the cosine similarities between the feature mapping of the training data and feature mapping of the smallest adversarial example starting at the test data

#test_loader: a data loader object at which we want to find the smallest adversarial example
#model: a neural net from which we extract the feature mappings
#attack: an attack with function signature attack(x,label,epsilon,model) which returns an adversarial example
#upper_bound_guess: an upper bound on the size of the smallest adversarial perturbation 
#strict: if True, upper_bound_guess is treated as a strict upper bound on the size of the smallest adversarial perturbation, otherwise it's assumed to be just a guess (setting this variable as true produces a significant speedup)
#error: stopping condition for the binary search-- the binary search terminates when upper_bound - lower bound <error
#fail_value: this value is returned if an adversarial example is not found
def cosine_similarities_at_smallest_perturbation(test_loader,model, feature_mapping, attack,upper_bound_guess=1,strict=False,error=10**-4,fail_value=None):
    cos_sims=[]
    for (x,label) in test_loader:
        radius,adversarial_example=smallest_adversarial_perturbation(x, label,model, attack, upper_bound_guess=upper_bound_guess,strict=strict,error=error,fail_value=fail_value)
        cos_sim=cosine_similarity(model,feature_mapping,adversarial_example)
        cos_sims.append(cos_sim)
    return np.array(cos_sims)


#a vectorized version of the cosine_similarities_at_smallest_perturbation function

def cosine_similarities_at_smallest_perturbation_vectorized(test_loader,model, feature_mapping, attack,upper_bound_guess=1,strict=False,error=10**-4,fail_value=None):
    xs=[]
    for (x,label) in test_loader:
        radius,adversarial_example=smallest_adversarial_perturbation(x, label,model, attack, upper_bound_guess=upper_bound_guess,strict=strict,error=error,fail_value=fail_value)
        xs.append((adversarial_example,label))
    ftrs=ntk_methods.feature_mapping(model ,xs,data_set=True,return_torch_tensor=False)
    cs=cosine_similarity_vectorized(ftrs,feature_mapping)
    cs=np.array(cs)
    #cs=np.expand_dims(cs,[1,2])
    return cs
  


#test_loader: a data loader containing the test set
#model: a neural net
#feature_mapping: a feature mapping of the training set
#attack: an adversarial attack with function signature attack(model,x,label) which returns and adversarial example
def cosine_similarities_of_attack(test_loader,model,feature_mapping,attack):
    cos_sims_correct=[]
    cos_sims_incorrect=[]
    for (x,label) in test_loader:
        adversarial_example=attack(model,x,label)
        cos_sim=cosine_similarity(model,feature_mapping,adversarial_example)
        adversarial_ys=model(adversarial_example)
        _, predicted = adversarial_ys.max(1)
        correct_prediction=predicted.eq(label).item()
            #add 1 to the running tally for correctly classified points if the correct class was predicted
        if correct_prediction:
            cos_sims_correct.append(cos_sim)
        else:
            cos_sims_incorrect.append(cos_sim)

    return cos_sims_correct, cos_sims_incorrect



#a vectorized version of cosine_similarities_of_attack
def cosine_similarities_of_attack_vectorized(test_loader,model,feature_mapping,attack):
    cos_sims_correct=[]
    cos_sims_incorrect=[]
    adversarial_examples=[]
    for (x,label) in test_loader:
        adversarial_example=attack(model,x,label)
        adversarial_examples.append((adversarial_example,label))
    #ftrs=ntk_methods.feature_mapping(model,adversarial_examples,data_set=True,return_torch_tensor=False)
    ftrs=ntk_methods.feature_mapping(model,adversarial_examples,data_set=True,return_torch_tensor=False)
    cs=cosine_similarity_vectorized(ftrs,feature_mapping)
    cos_sims_correct=[]
    cos_sims_incorrect=[]
    for i in range(len(adversarial_examples)):
        tpl=adversarial_examples[i]
        adversarial_example=tpl[0]
        label=tpl[1]
        adversarial_ys=model(adversarial_example)
        _, predicted = adversarial_ys.max(1)
        correct_prediction=predicted.eq(label).item()
        cos_sim=cs[i]
            #add 1 to the running tally for correctly classified points if the correct class was predicted
        if correct_prediction:
            cos_sims_correct.append(cos_sim)
        else:
            cos_sims_incorrect.append(cos_sim)

    return np.array(cos_sims_correct), np.array(cos_sims_incorrect)
    




