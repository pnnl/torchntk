#for training 10 trials of a fully connected 3-layer network to classify handwritten digits (MNIST dataset)
import pickle
import adversaries
import adversarial_training
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import torch.optim as optim
import time
import os
import platform
import argparse






#defining the neural net
class Net_sigmoid(nn.Module):
    def __init__(self):
        super(Net_sigmoid, self).__init__()
        self.fc1 = nn.Linear(28*28,100 )
        self.fc2 = nn.Linear(100,100 )
        self.fc3 = nn.Linear(100,100 )
        #self.fc4 = nn.Linear(100,100 )
        self.fc4=  nn.Linear(100,2)

    def forward(self, x):
        sigmoid=nn.Sigmoid()
        x = sigmoid(self.fc1(torch.flatten(x,start_dim=1)))
        x = sigmoid(self.fc2(x))
        x = sigmoid(self.fc3(x))
        #x = sigmoid(self.fc4(x))
        x=  self.fc4(x)
        return x
        #return F.log_softmax(x, dim=1)



#training the nets
def main():
    #'trial name' is an input from the command line, so that 10 trials can be trained in parallel
    parser=argparse.ArgumentParser(description="trial name")
    parser.add_argument('trial_name',metavar='trl',type=ascii,nargs=1,help='name of the trial run' )
    try:
        args=parser.parse_args()
        trial_name=args.trial_name[0]
    except (SystemExit, TypeError) as error:
        trial_name="default_trial"
    epochs=100
    eps_list=[0, 0.05, 0.1, 0.15, 0.20, 0.25,.30]#a neural net is adversarially trained for every perturbation radius in this list

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    mnist_train_set= datasets.MNIST('../data', train=True, download=True, transform=transforms.Compose([
        transforms.ToTensor(),]))
    mnist_train_sbset=[]

    #the net is only trained on the 1,7 instances of MNIST
    for (data, label) in mnist_train_set:
        if label==1:
            mnist_train_sbset.append((data,label))
        elif label==7:
            mnist_train_sbset.append((data,0))
    for epsilon in eps_list:
        seed= int(time.time())
        torch.manual_seed(seed)
        start=time.time()
        train_loader = torch.utils.data.DataLoader(mnist_train_sbset,batch_size=64, shuffle=True)
 
        model=Net_sigmoid()
        learning_rate=.0001
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate)


        loss_function=F.cross_entropy

        #defining the adversary
        p=float('inf')
        iter=7
        if epsilon!=0:
            def adversary(model,x,y):
                return adversaries.pgd_attack_p(x,y,epsilon,model,loss_function,p,iter,device=device,rand_init=True)
            adv_name='pgd'+str(iter)
        else:
            adversary=adversaries.trivial_attack
            adv_name='None'
  
        train_error_list, classification_error_list=adversarial_training.classic_adversarial_training(model, epochs, train_loader, adversary, optimizer, loss_function,1,device=device)
        variables_dict={"epochs":epochs,"epsilon":epsilon,"adversary":adv_name,"loss_function":loss_function,"iterations":iter,"p":p,"digits":(7,1),"training_error":train_error_list,"classification_error":classification_error_list,"random_seed": seed}
        
        
        # Parent Directory path
        parent_parent_dir = "models_final"
        parent_dir="binary_sigmoid_models"
        # Directory

        #creating files if they don't already exist
        if not os.path.isdir(parent_parent_dir):
            os.mkdir(parent_parent_dir)
        path = os.path.join(parent_parent_dir,parent_dir)
        if not os.path.isdir(path):
            os.mkdir(path)
        prefix=os.path.join(path,trial_name)
        if not os.path.isdir(prefix):
            os.mkdir(prefix)
        prefix=os.path.join(prefix,trial_name+"_epsilon="+str(epsilon))

  


        file=open(prefix+"_hyperparameters.p",'wb')
        pickle.dump(variables_dict,file)
        file.close()

        torch.save(model.state_dict(),prefix+"model.pt")
        torch.save(optimizer.state_dict(),prefix+"optimizer.pt")
        end=time.time()
        print(end-start, "\n")



if __name__ == "__main__":
    if platform.platform() == "Windows-10-10.0.19042-SP0":
        os.chdir("C:\\Users\\fran316\\OneDrive - PNNL\\Documents\\experiments")
    else:
        os.chdir("/people/fran316/experiments")
    main()
