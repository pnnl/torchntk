import smallest_adversarial_example
import adversaries
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import pickle
import numpy as np
import platform
import ntk_methods

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



def smallest_adversarial_perturbation_test():
    i=0
    dset= datasets.MNIST('../pretrained_data', train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            ]))

    data,target=dset[0]

    pretrained_model = "pretrained_models/lenet_mnist_model.pth"

    # Initialize the network
    model = Net().to('cpu')

    # Load the pretrained model
    model.load_state_dict(torch.load(pretrained_model, map_location='cpu'))

    # Set the model in evaluation mode. In this case this is for the Dropout layers
    model.eval()
    p=2
    iter=5
    def fgsm_adversary(x, label, epsilon,model):
        return adversaries.fgsm_attack_p(x,label,epsilon,model,loss_function,p)

    def pgd_adversary(x,label,epsilon,model):
        return adversaries.pgd_attack_p(x,label,epsilon,model,loss_function,p,iter)
    
    advs=[fgsm_adversary, pgd_adversary]
    loss_function=F.nll_loss
    data.requires_grad=True
    for adv in advs:
        radius,example=smallest_adversarial_example.smallest_adversarial_perturbation(data, target,model, adv)
        pred=model(example)
        _, predicted = pred.max(1)
        correct=predicted.eq(target).item()
        example2=adv(data,target,radius-10**-4,model)
        pred2=model(example2)
        _, predicted2 = pred2.max(1)
        correct2=predicted2.eq(target).item()
        print([not correct, correct2])




def attribute_histogram_test():
    test_loader_start= datasets.MNIST('../pretrained_data', train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            ]))
    sbset=list(range(0,2))
    sbset_1=torch.utils.data.Subset(test_loader_start,sbset)
    test_loader=torch.utils.data.DataLoader(sbset_1,batch_size=1)
    #test_loader=torch.utils.data.DataLoader(test_loader_start)
    train_loader_start=datasets.MNIST('../pretrained_data', train=True, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            ]))
    sbset=list(range(0,2))
    sbset_1=torch.utils.data.Subset(train_loader_start,sbset)
    train_loader=torch.utils.data.DataLoader(sbset_1,batch_size=1)
    #train_loader=torch.utils.data.DataLoader(train_loader_start)
    loss_function=F.nll_loss;
    p=float('inf')
    def attack(x,label,epsilon,model):
        return adversaries.fgsm_attack_p(x,label,epsilon,model,loss_function,p)

    iterations=20
    repetitions=10
    def pgd_adversary(x,label,epsilon,model):
        return adversaries.pgd_repetitions(x,label,epsilon,model,loss_function,p,iterations,repetitions=repetitions,include_center_initialization=True)

    def attribute(model, tpl):
        x=tpl[0]
        label=tpl[1]
        epsilon,example= smallest_adversarial_example.smallest_adversarial_perturbation(x,label,model,attack,strict=True)
        return (epsilon,)
    def attribute_cossim(model,tpl):
        x=tpl[0]
        label=tpl[1]
        epsilon,example= smallest_adversarial_example.smallest_adversarial_perturbation(x,label,model,pgd_adversary)
        ntk_load=np.load('temporary\\MNIST_adversarial\\NTK_dict.npy',allow_pickle=True).item()
        ntk=[ ntk_load[key][1:10000,1:10000] for key in ntk_load.keys()]
        
    
    pretrained_model = "pretrained_models/lenet_mnist_model.pth"

    # Initialize the network
    model1 = Net().to('cpu')

    # Load the pretrained model
    model1.load_state_dict(torch.load(pretrained_model, map_location='cpu'))

    # Set the model in evaluation mode. In this case this is for the Dropout layers
    model1.eval()
    
    def model2(x):
        pred=model1(x);
        pred[0]=2*pred[0]
        return pred

    model_list=[model1]

    #file_name="temporary_pickles\\perturbation_radius.p"
    #smallest_adversarial_example.calculate_attribute(model_list,test_loader,attribute,file_names=[file_name])
    #file=open(file_name,'rb')
    #values_list=pickle.load(file)
    file_name_base="MNIST_all"
    file_name=os.path.join("temporary_pickles",file_name_base)
    feature_mapping_list=[]
    for model in model_list:
        ntk=ntk_methods.feature_mapping(model,train_loader)
        feature_mapping_list.append(ntk.detach().numpy())
    smallest_adversarial_example.calculate_smallest_perturbations_and_cossim( test_loader,model_list, feature_mapping_list,pgd_adversary,file_name=file_name,strict=False)
    file=open(file_name+'_cossim.p', 'rb')
    values_list=pickle.load(file)
    figure_pth=os.path.join("plots",file_name_base)
    smallest_adversarial_example.attribute_histogram(values_list,save_file=figure_pth,rng=(0,1),bins=None,x_label="cosine similarity")

    


if __name__ == "__main__":
    if platform.platform() == "Windows-10-10.0.19042-SP0":
        os.chdir("C:\\Users\\fran316\\OneDrive - PNNL\\Documents\\experiments")
    else:
        os.chdir("/people/fran316/experiments")
    smallest_adversarial_perturbation_test()


