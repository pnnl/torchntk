import adversaries
import torch
import pickle
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
import os
from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.torch.attacks.projected_gradient_descent import (
    projected_gradient_descent,
)


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






#tests pgd by comparing to previously pickled fgsm runs
#compare=True comparres to previously saved data, compare=False overrides the previously saved data
def pgd_test(compare=True,boolean_output=True):
    torch.manual_seed(1)
    i=0
    dset= datasets.MNIST('../pretrained_data', train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            ]))

    data,target=dset[0]

    pretrained_model = "pretrained_data/lenet_mnist_model.pth"

    # Initialize the network
    model = Net().to('cpu')

    # Load the pretrained model
    model.load_state_dict(torch.load(pretrained_model, map_location='cpu'))

    # Set the model in evaluation mode. In this case this is for the Dropout layers
    model.eval()


    p_list=[2,float('inf')]
    rand_init=[True,False]
    targeted_list=[True,False]
    #labels=[ torch.tensor([1]),torch.tensor([7])]
    labels=[1,7]
    intermediate_projections=[True,False]
    number_comparisons=2**5
    epsilon=3
   
    loss_function = F.nll_loss
    iter=20
    data.requires_grad=True
    adversarial_points=[adversaries.pgd_attack_p(data,label,epsilon,model,loss_function,p,iter,rand_init=init,clip_min=0,clip_max=1,intermediate_clip=proj,targeted=targeted) for p in p_list for init in rand_init for targeted in targeted_list for label in labels for proj in intermediate_projections]
    
    if compare:
        file=open("testing_pickles/pgd_test.p", "rb")
        comparison_data=pickle.load(file)
        file.close()
        if boolean_output:
            comparisons=[False]*number_comparisons
            for i in range(number_comparisons):
                comparisons[i]= torch.all(comparison_data[i].eq(adversarial_points[i])).item()
            print(all(comparisons))
        else:
            comparisons=[torch.dist(adversarial_points[i],comparison_data[i]).item() for i in range(number_comparisons)]
            print(comparisons)
    else:
        file=open("testing_pickles/pgd_test.p", "wb")
        pickle.dump(adversarial_points,file)
        file.close()


#tests pgd by comparing to cleverhans
def pgd_cleverhans_test():
    i=0
    dset= datasets.MNIST('../pretrained_data', train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            ]))

    data,target=dset[0]

    pretrained_model = "pretrained_data/lenet_mnist_model.pth"

    # Initialize the network
    model = Net().to('cpu')

    # Load the pretrained model
    model.load_state_dict(torch.load(pretrained_model, map_location='cpu'))

    # Set the model in evaluation mode. In this case this is for the Dropout layers
    model.eval()


    p_list=[2,float('inf')]
    targeted_list=[True,False]
    labels=[ torch.tensor([1]),torch.tensor([7])]
    labels2=[1,7]
    number_comparisons=2**3
    epsilon=1
   
    loss_function = F.nll_loss
    iter=20
    step_size=2*epsilon/iter*1.4
    torch.manual_seed(1)
    data.requires_grad=True
    adversarial_points=[adversaries.pgd_attack_p(data,label,epsilon,model,loss_function,p,iter,rand_init=False,clip_min=0,clip_max=1,intermediate_clip=True,targeted=targeted) for p in p_list for targeted in targeted_list for label in labels2]
    torch.manual_seed(1)
    cleverhans_points=[projected_gradient_descent(model,data,epsilon,step_size,iter,p,clip_min=0,clip_max=1,y=label,targeted=tgd,rand_init=False) for p in p_list  for tgd in targeted_list for label in labels]
    comparison2=[torch.dist(adversarial_points[i],cleverhans_points[i],2).item() for i in range(number_comparisons)]
    print(comparison2)

#tests fgsm by comparing to previously pickled fgsm runs
#compare=True comparres to previously saved data, compare=False overrides the previously saved data
def fgsm_test(compare=True,boolean_output=True):
    i=0
    dset= datasets.MNIST('../pretrained_data', train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            ]))

    data,target=dset[0]
    #target=torch.tensor([torch.tensor(target)])

    pretrained_model = "pretrained_data/lenet_mnist_model.pth"

    # Initialize the network
    model = Net().to('cpu')

    # Load the pretrained model
    model.load_state_dict(torch.load(pretrained_model, map_location='cpu'))

    # Set the model in evaluation mode. In this case this is for the Dropout layers
    model.eval()


    p_list=[1.5,2,3,10,float('inf')]
    epsilon=3
    loss_function=F.nll_loss
    #data.requires_grad=True
    adversarial_points=[adversaries.fgsm_attack_p(data,target,epsilon,model,loss_function,p) for p in p_list]
    if compare:
        file=open("testing_pickles/fgsm_test.p", "rb")
        comparison_data=pickle.load(file)
        file.close()
        if boolean_output:
            comparisons=[False]*len(p_list)
            for i in range(len(p_list)):
                comparisons[i]= torch.all(comparison_data[i].eq(adversarial_points[i])).item()
            print(all(comparisons))
        else:
            comparisons=[torch.dist(adversarial_points[i],comparison_data[i]).item() for i in range(len(p_list))]
            print(comparisons)
    else:
        file=open("testing_pickles/fgsm_test.p", "wb")
        pickle.dump(adversarial_points,file)
        file.close()



#tests fgsm by comparing to cleverhans
def fgsm_cleverhans_test():
    i=0
    dset= datasets.MNIST('../pretrained_data', train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            ]))

    data,target=dset[0]

    pretrained_model = "pretrained_data/lenet_mnist_model.pth"

    # Initialize the network
    model = Net().to('cpu')

    # Load the pretrained model
    model.load_state_dict(torch.load(pretrained_model, map_location='cpu'))

    # Set the model in evaluation mode. In this case this is for the Dropout layers
    model.eval()


    p_list=[2,float('inf')]
    epsilon=3

    loss_function=F.nll_loss
    #data.requires_grad=True;
    my_adversarial_points=[adversaries.fgsm_attack_p(data,target,epsilon,model,loss_function,p) for p in p_list]

    
    y=model(data);

    label=torch.tensor([target])
    loss = F.nll_loss(y,label)
    loss.backward();

    clever_hans_adversarial_points=[fast_gradient_method(model,data,epsilon,p) for p in p_list]
    #comparison=[torch.all(my_adversarial_points[i].eq(clever_hans_adversarial_points[i])).item() for i in range(len(p_list))]
    comparison2=[torch.dist(my_adversarial_points[i],clever_hans_adversarial_points[i],p).item() for p in p_list]
    print(comparison2)


def nan_test():
    file=open("zzz-nan_test.p","rb")
    v=pickle.load(file)
    file.close()
    model=v[0]
    inputs=v[1]
    labels=v[2]

    epsilon=3
    loss_function=F.nll_loss
    p=2
    iter=2
    def input_attack(model,x,y):
        return adversaries.fgsm_attack_p(x,y,epsilon,model,loss_function,p)
    def attack(model,xs,ys):
            return adversaries.vectorize_adversary(model, xs,ys,input_attack)
    vv=attack(model,inputs,labels)
    t=1
    x=inputs[0]
    for_andrew=[model,x]
    file=open("zzz-pickle_for_andrew.p","wb")
    pickle.dump(for_andrew,file)
    file.close()

if __name__ == "__main__":
    os.chdir("C:\\Users\\fran316\\OneDrive - PNNL\\Documents\\experiments")
    #fgsm_test(compare=True,boolean_output=False)
    #fgsm_cleverhans_test()
    pgd_test(compare=True,boolean_output=False)
    pgd_cleverhans_test()
    #nan_test()
