import adversaries
import adversarial_training
import torch
import pickle
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
import os
import torch.optim as optim
import train_binary_MNIST_sigmoid_networks
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
    
def classic_adversarial_training_test(compare=True,boolean_output=True):
    train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            ])),
        batch_size=3, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            ])),
        batch_size=1, shuffle=True) 
    model=Net()
    learning_rate=.1
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)


    epochs=1
    epsilon=3
    loss_function=F.nll_loss
    p=2
    iter=2
    def adversary_fgsm(model,x,y):
        #x=x+torch.zeros_like(x)
        #x.retain_grad()
        return adversaries.fgsm_attack_p(x,y,epsilon,model,loss_function,p)
    def adversary_pgd(model,x,y):
        return adversaries.pgd_attack_p(x,y,epsilon,model,loss_function,p,iter)
    trivial_adversary=adversaries.trivial_attack
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #print(test( model, test_loader, adversary, loss_function, device,adv_classification_risk=False))
    torch.manual_seed(1)
    train_error_list=[0]*2
    classification_error_list=[0]*2
    
    train_error_list[0], classification_error_list[0]=adversarial_training.classic_adversarial_training(model, epochs, test_loader, adversary_fgsm, optimizer, loss_function,1,device)
    train_error_list[1], classification_error_list[1]=adversarial_training.classic_adversarial_training(model, epochs, test_loader, adversary_pgd, optimizer, loss_function,1,device)
    if compare:
        lst=[train_error_list]
        file=open("testing_pickles/classic_adversarial_training_test.p", "rb")
        te_comp,ce_comp=pickle.load(file)
        file.close()
        te_comp=np.array(te_comp)
        ce_comp=np.array(ce_comp)
        train_error_list=np.array(train_error_list)
        classification_error_list=np.array(classification_error_list)
        if boolean_output:
            print([np.equal(classification_error_list,ce_comp), np.equal(train_error_list, te_comp)])
        else:
            print([np.linalg.norm(classification_error_list-ce_comp), np.linalg.norm(train_error_list- te_comp)])
        
        
    else:
        lst=[train_error_list,classification_error_list]
        file=open("testing_pickles/classic_adversarial_training_test.p", "wb")
        pickle.dump(lst,file)
        file.close()


def test_function_test(compare=True, boolean_output=True):
    test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            ])),
        batch_size=1, shuffle=True) 
    model=Net()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #print(test( model, test_loader, adversary, loss_function, device,adv_classification_risk=False))
    torch.manual_seed(1)
    train_error_list=[0]*2
    classification_error_list=[0]*2
    loss_function=F.nll_loss
    epsilon=.3
    p=float('inf')
    iter=2
    def adversary_fgsm(model,x,y):
        #x=x+torch.zeros_like(x)
        #x.retain_grad()
        return adversaries.fgsm_attack_p(x,y,epsilon,model,loss_function,p)
    def adversary_pgd(model,x,y):
        #x=x+torch.zeros_like(x)
        return adversaries.pgd_attack_p(x,y,epsilon,model,loss_function,p,iter)
    train_error_list[0],classification_error_list[0]=adversarial_training.test(model,test_loader,adversary_fgsm,loss_function,device)
    train_error_list[1],classification_error_list[1]=adversarial_training.test(model,test_loader,adversary_pgd,loss_function,device)
    if compare:
        lst=[train_error_list]
        file=open("testing_pickles/adversarial_test.p", "rb")
        te_comp,ce_comp=pickle.load(file)
        file.close()
        te_comp=np.array(te_comp)
        ce_comp=np.array(ce_comp)
        train_error_list=np.array(train_error_list)
        classification_error_list=np.array(classification_error_list)
        if boolean_output:
            print([np.equal(classification_error_list,ce_comp), np.equal(train_error_list, te_comp)])
        else:
            print([np.linalg.norm(classification_error_list-ce_comp), np.linalg.norm(train_error_list- te_comp)])
        
        
    else:
        lst=[train_error_list,classification_error_list]
        file=open("testing_pickles/adversarial_test_test.p", "wb")
        pickle.dump(lst,file)
        file.close()
 


#TODO: this test throws an error, need to change train_loader and test_loader so that it is binary classification
def small_test_train_networks():
    model=train_binary_MNIST_sigmoid_networks.Net_sigmoid()
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True, transform=transforms.Compose([
                transforms.ToTensor(),
                ])),
            batch_size=1, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, download=True, transform=transforms.Compose([
                transforms.ToTensor(),
                ])),
            batch_size=1, shuffle=True)
    learning_rate=.001
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)


    loss_function=F.cross_entropy
    p=float('inf')
    adversary=adversaries.trivial_attack
    class_error_start=adversarial_training.test(model,test_loader,adversary,loss_function)
    print('initial classification error:'+str(class_error_start))
    epochs=1
    train_error_list, classification_error_list=adversarial_training.classic_adversarial_training(model, epochs, train_loader, adversary, optimizer, loss_function,1,device='cpu')
    class_error_end=adversarial_training.test(model,test_loader,adversary,loss_function)
    print('initial classification error:'+str(class_error_end))


if __name__ == "__main__":
    os.chdir("C:\\Users\\fran316\\OneDrive - PNNL\\Documents\\experiments")
    #classic_adversarial_training_test(compare=True,boolean_output=False)
    #test_function_test(compare=True, boolean_output=True)
    small_test_train_networks()
    
