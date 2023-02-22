#Implements methods involving the feature mappings of a neural net

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.svm import SVC


#returns the concatenation of each output neuron differentiated in terms of each model parameter evaluated on data

#net: the neural net for which we compute the feature mapping
#data: the input data at which we compute the feature mapping
#data_set: true if 'data' is a data set, false if it is a tensor. The 0th dimension is assumed to be the batch dimension
#higher_derivatives: True if one will want to compute higher order derivatives in the future
#return_torch_tensor: returns a torch tensor if True, and a numpy array if false. (Implemented separately for efficiency with memory.)
def feature_mapping(net, data, data_set=True, higher_derivatives=False,return_torch_tensor=True):
    if return_torch_tensor:
        derivative_list=[]
        for dt in data:
            if data_set:
                data_point=dt[0]
            else:
                data_point=dt
            #x=data_point.detach()
            x=data_point
            y_list=net(x)[0]
            w_derivative_list=[]
            for i in range(y_list.size()[0]):
                y=y_list[i]
                params=net.parameters()
                for p in params:
                    if higher_derivatives:
                        p_grad=torch.autograd.grad(y,p,create_graph=True,allow_unused=True)[0]
                    else:
                        p_grad=torch.autograd.grad(y,p,retain_graph=True)[0]
                    w_derivative_list.append(torch.flatten(p_grad))
            
            feature =torch.cat(w_derivative_list)
            derivative_list.append(feature)
        feature_mapping=torch.stack(derivative_list)
        feature_mapping=torch.transpose(feature_mapping,0,1)
        return feature_mapping
    else:
        #calculate the length of the feature mapping
        dt=data[0]
        if data_set:
            data_point=dt[0]
        else:
            data_point=dt
        #x=data_point.detach()
        x=data_point
        feature=get_feature_single_derivatives(net,x)
        feature_length=feature.shape[0]
        #calculate the number of data points
        num_data_points=0
        for dt in data:
            num_data_points+=1
        #pre-allocate for np-array
        feature_mapping=np.zeros((feature_length,num_data_points), np.float32)
        for i in range(num_data_points):
            dt=data[i]
            if data_set:
                data_point=dt[0]
            else:
                data_point=dt
            feature=get_feature_single_derivatives(net,data_point).numpy()
            feature_mapping[:,i]=feature
        return feature_mapping
    
#returns the feature mapping of a single data point x. Assumes that higher order derivatives will not be calculated

#net: the neural net for which we compute the feature mapping
#x: a single data point
#TODO to avoid repeated code, add in the case for highter order derivatives to this method
def get_feature_single_derivatives(net,x):
    y_list=net(x)[0]
    w_derivative_list=[]
    for i in range(y_list.size()[0]):
        y=y_list[i]
        params=net.parameters()
        for p in params:
            p_grad=torch.autograd.grad(y,p,retain_graph=True)[0]
            w_derivative_list.append(torch.flatten(p_grad))
    feature =torch.cat(w_derivative_list)
    return feature


#returns the NTK of the neural net evaluated on input data

#net: a neural net, assumed to extend nn.Module
#data: a dataset object or a list of (x,label) tuples
def total_ntk(net, data):
    ftr_mapping=feature_mapping(net,data).numpy()
    krn=np.dot(ftr_mapping.transpose(),ftr_mapping)
    return krn


#returns an sklearn SVM trained using a feature mapping

#train_feature_mapping: the feature mapping for computing the SVM (intended to be the feature_mapping function evaluated on a neural net and the training set)
#Y: the labels
#C: the regularization parameter
def svm_from_kernel_matrix(train_feature_mapping,Y,C=1.0):
    ftr_mapping=train_feature_mapping.detach().numpy()
    krn=np.dot(ftr_mapping.transpose(),ftr_mapping)
    svm_fit=SVC(C=C, kernel='precomputed')
    

    svm_fit.fit(krn,Y)
    return svm_fit







#predicts the label of data

#net: a neural net
#test_loader: a data set of list of (x,label) tuples
#train_feature_mapping: the feature mapping of the training data according to the NTK (i.e., the output of the feature_mapping function evaluated on net and the training set)
#svm: an sklearn SVM using train_feature_mapping along with the labels of the training data
def svm_predict(svm,net, train_feature_mapping, test_loader):
    ftr=feature_mapping(net, test_loader).numpy()
    inner_prod=np.dot(np.transpose(ftr),train_feature_mapping)
    return svm.predict(inner_prod)


    
    














