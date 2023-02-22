#implements an SVM using pytorch so that it can be attacked using gradient-based methods
import torch.nn as nn
import torch
import ntk_methods
import numpy as np



class binary_torch_SVM(nn.Module):
        #net: the neural net to be approximated using the SVM
        #svm: an sklearn svm trained using the NTK of net on the training data along with the correct data labels 
        #feature mapping: the jacobian net evaluated on the training set

        #sets the coefficients of the SVM
        def __init__(self, net,svm,feature_mapping):
            super(binary_torch_SVM, self).__init__()
            self.net=net
            self.intercept=torch.tensor(svm.intercept_)
            support_indices=svm.support_
            sz=feature_mapping.shape
            self.coef=torch.zeros([sz[0], 1])
            dual_coefs=svm.dual_coef_[0]
            for j in range(len(support_indices)):
                support_index=support_indices[j]
                ft=feature_mapping[:,support_index]
                ft=ft[:,None]
                self.coef+=dual_coefs[j]*ft


        #xs: a torch tensor containing points to be classified

        #finds the feature mapping of each data point x in xs and outputs w*x+b, where b is the intercept and w the coefficients of the slopes of the neural net  
        def forward(self,xs):
            features=ntk_methods.feature_mapping(self.net,xs,data_set=False,higher_derivatives=True)
            tnsr=[torch.matmul(feature_vector,self.coef)+self.intercept for feature_vector in torch.transpose(features,0,1)]
            tnsr=torch.stack(tnsr)

            return tnsr 


        #predicts the class of a list or tensor of x values
        #outputs 0 or 1
        def predict(self,x_list):
            y_list=[0]*len(x_list)
            vals=self.forward(x_list)
            for i in range(len(x_list)):
                val=vals[i]
                if val.item()>0:
                    y_list[i]=1
            return np.array(y_list)

            
            