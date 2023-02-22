#methods for adversarially training and testing neural nets


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import adversaries
import math
from torchvision import datasets, transforms
import os


#trains a neural net using the adversarial training algorithm

#model: the neural net for training, assumed to extend nn.Module
#epochs: the number of training epochs
#data_loader: data loader for looping over the data
#input_attack: the adversarial attack. input_attack must have inputs (model,x,y) and must output a torch tensor the same dimensions as x as an attack image. input_attack can be vectorized or unvectorized, but the vectorized_adversary field must be set accordingly
#optimizer: the optimizer for training
#loss_function: the loss function to be minimized during training
#epoch_print_frequency (optional): training progress will be printed every epoch_print_frequency epochs
#device: device for training. If "use_all_GPUs" is true, all GPUs will be used instead of the specified device
#x_gradient_required (optional): whether or not "input attack" requires gradients of the model in x
#vectorized_adversary (optional): whether the function "input_attack" is vectorized. If the batch size is greater than 1 and vectorized_adversary is false, the method will apply the adversary to the data points in a loop
def classic_adversarial_training(model, epochs, data_loader, input_attack, optimizer, loss_function,epoch_print_frequency=10,device=torch.device("cpu"),x_gradient_required=True,vectorized_adversary=False):
    model.to(device)
    model.train()
    if vectorized_adversary or data_loader.batch_size==1:
        attack=input_attack
    else:
        def attack(model,xs,ys):
            return adversaries.vectorize_adversary(model, xs,ys,input_attack)
    classification_error_list=[]
    train_error_list=[]
    for epoch in range(epochs):
        training_risk=0
        correct=0
        total=0
        for batch, (inputs, targets) in enumerate(data_loader):
            #move data to device
            inputs=inputs.to(device)
            targets=targets.to(device)
            #zeroing the gradients
            optimizer.zero_grad()
            
            
            #requiring gradients in the x-variable
            if x_gradient_required:
                inputs.requires_grad=True #TODO test this if statement
            #inputs.to(device)
            #print(inputs.get_device())
            adversarial_examples=attack(model,inputs,targets)#generating the adversarial examples
            adversarial_ys=model(adversarial_examples)#running the adversarial examples through the model
            loss=loss_function(adversarial_ys,targets)#evaluating the loss


            
            #calculating the gradients
            loss.backward()

            #changing the neural net
            optimizer.step()

            #add the loss of the current to the current tally for the risk
            training_risk+=loss.sum().item()

            #counting the number of data points
            total+=targets.size(0)

            #finding the class predictions
            _, predicted = adversarial_ys.max(1)

            #add 1 to the running tally for correctly classified points if the correct class was predicted
            correct += predicted.eq(targets).sum().item()
        #normalizing the empirical risks
        adversarial_classification_correct=correct/total
        training_risk=training_risk/total
        #printing progress
        if epoch%epoch_print_frequency==0:
            print("Epoch:", epoch, " Training risk:", training_risk, "Adversarial classification error:", 1-adversarial_classification_correct)
            train_error_list.append(training_risk)
            classification_error_list.append(1-adversarial_classification_correct)
    model.eval()
    return train_error_list, classification_error_list



#calculates the test error of a neural net under adversarial perturbations

#model: neural net under attack, assumed to extend nn.Module
#test_loader: for loading the test set. Assumes a batch size of 1
#attack: function that takes as input (model,x,label) and outputs and adversarial example
#loss_function: loss function for evaluating model on data
#device: TODO currently not implemented, the device on which the model is tested
#max prediction: the label is predicted according to the maximum of the last output layer of model if True, and according to the minimum if false
def test( model, test_loader, attack, loss_function, device="cpu",max_prediction=True):
    class_risk_sum=0
    train_risk_sum=0
    total_data_points=0
    for x,label in test_loader:
        #adding 1 to the total data points
        total_data_points+=1
        # TODO moving the data and lable to the device
        #x, label = x.to(device), label.to(device)
        #zero the gradients in the model
        model.zero_grad()
        #require gradients on x, important for adversarial examples
        x.requires_grad=True
        #generating the adversarial example
        adversarial_example=attack(model,x,label)
        #running the adversarial example through the model:
        adversarial_y=model(adversarial_example)
        #evaluating the loss:
        #finding the class predictions
        if max_prediction:
            _, predicted = adversarial_y.max(1)
        else:
            _, predicted = adversarial_y.min(1)
            #add 1 to the running tally for correctly classified points if the correct class was predicted
        class_risk_sum += (~predicted.eq(label)).item()
        #evaluating the adversarial loss and adding to the running tally
        loss=loss_function(adversarial_y,label)
        train_risk_sum+=loss.item()
            
    #calculating the risk
    class_risk=class_risk_sum/total_data_points
    train_risk=train_risk_sum/total_data_points
    return train_risk, class_risk




