import ntk_methods
import os
import torch.nn as nn
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
import platform
import binary_torch_svm






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

def svm_performance_test():
    pretrained_model = "pretrained_models/lenet_mnist_model.pth"

    # Initialize the network
    model1 = Net().to('cpu')

    # Load the pretrained model
    model1.load_state_dict(torch.load(pretrained_model, map_location='cpu'))

    # Set the model in evaluation mode. In this case this is for the Dropout layers
    model1.eval()

    train_loader_start=datasets.MNIST('../pretrained_data', train=True, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            ]))
    sbset=list(range(0,100))
    sbset_1=torch.utils.data.Subset(train_loader_start,sbset)
    train_loader=torch.utils.data.DataLoader(sbset_1,batch_size=1)
    #train_loader=torch.utils.data.DataLoader(train_loader_start)



    test_loader_start= datasets.MNIST('../pretrained_data', train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            ]))
    sbset=list(range(0,2))
    sbset_1=torch.utils.data.Subset(test_loader_start,sbset)
    test_loader=torch.utils.data.DataLoader(sbset_1,batch_size=1)
    dt=ntk_methods.feature_mapping(model1,test_loader).numpy()

    train_features=ntk_methods.feature_mapping(model1,train_loader)
   #krn=np.dot(train_features.transpose(),train_features)
    Y=[tpl[1].item() for tpl in train_loader]
    svm=ntk_methods.svm_from_kernel_matrix(train_features,Y)

    print(ntk_methods.svm_predict(svm,model1,train_features,test_loader))
    print([tpl[1] for tpl in test_loader])


def svm_sklearn_comparison():
    pretrained_model = "pretrained_models/lenet_mnist_model.pth"
       # Initialize the network
    model1 = Net().to('cpu')

    # Load the pretrained model
    model1.load_state_dict(torch.load(pretrained_model, map_location='cpu'))

    # Set the model in evaluation mode. In this case this is for the Dropout layers
    model1.eval()

    mnist_train_set= datasets.MNIST('../data', train=True, download=True, transform=transforms.Compose([
    transforms.ToTensor(),]))
    mnist_train_sbset=[]
    for (data, label) in mnist_train_set:
        if label==1:
            mnist_train_sbset.append((data,label))
        elif label==7:
            mnist_train_sbset.append((data,0))


    sbset=list(range(0,100))
    sbset_1=torch.utils.data.Subset(mnist_train_sbset,sbset)
    train_loader=torch.utils.data.DataLoader(sbset_1,batch_size=1,shuffle=False)
    #train_loader=sbset_1
    dt=ntk_methods.feature_mapping(model1,train_loader).numpy()




    train_features=ntk_methods.feature_mapping(model1,train_loader)
    #krn=np.dot(train_features.transpose(),train_features)
    Y=[tpl[1].item() for tpl in train_loader]
    svm=ntk_methods.svm_from_kernel_matrix(train_features,Y)
    torch_svm=binary_torch_svm.binary_torch_SVM(model1, svm,train_features)

    mnist_test_set= datasets.MNIST('../data', train=False, download=True, transform=transforms.Compose([
    transforms.ToTensor(),]))
    mnist_test_sbset=[]
    for (data, label) in mnist_test_set:
        if label==1:
            mnist_test_sbset.append((data,label))
        elif label==7:
            mnist_test_sbset.append((data,0))



    sbset=list(range(0,50))
    sbset_1=torch.utils.data.Subset(mnist_test_sbset,sbset)
    x_sbset_1=[x for (x,y) in sbset_1]
    test_loader=torch.utils.data.DataLoader(sbset_1,batch_size=1)
    sk_learn_predictions=ntk_methods.svm_predict(svm,model1,train_features,test_loader)
    torch_predictions=torch_svm.predict(x_sbset_1)
    #print(sk_learn_predictions)
    #print(torch_predictions)
    print(all(sk_learn_predictions==torch_predictions))

if __name__ == "__main__":
    if platform.platform() == "Windows-10-10.0.19042-SP0":
        os.chdir("C:\\Users\\fran316\\OneDrive - PNNL\\Documents\\experiments")
    else:
        os.chdir("/people/fran316/experiments")
    svm_sklearn_comparison()
    svm_performance_test()