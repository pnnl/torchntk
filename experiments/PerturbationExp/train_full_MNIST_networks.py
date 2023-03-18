#for training 10 trials of a fully connected 3-layer network to classify handwritten digits (MNIST dataset)
import adversaries
import adversarial_training
import torch
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



#the neural net class
class Net_relu(nn.Module):
    def __init__(self):
        super(Net_relu, self).__init__()
        self.fc1 = nn.Linear(28*28,100 )
        self.fc2 = nn.Linear(100,100 )
        self.fc3 = nn.Linear(100,100 )
        #self.fc4 = nn.Linear(100,100 )
        self.fc4=  nn.Linear(100,10)

    def forward(self, x):
        x = F.relu(self.fc1(torch.flatten(x,start_dim=1)))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        #x = F.relu(self.fc4(x))
        x=  self.fc4(x)

        return x

#kaiming normal initialization for the weights
def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight,a=0.01)
        m.bias.data.fill_(0.0)
        pass

#train and save 10 networks
def main():
    epochs=100
    trial_number=10

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    mnist_train_set= datasets.MNIST('../data', train=True, download=True, transform=transforms.Compose([
        transforms.ToTensor(),]))
    for trl in range(trial_number):
        seed= int(time.time())
        torch.manual_seed(seed)
        start=time.time()
        train_loader = torch.utils.data.DataLoader(mnist_train_set,batch_size=64, shuffle=True)
 
        model=Net_relu()
        model.apply(init_weights)
        learning_rate=.001
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate)


        loss_function=F.cross_entropy

        adversary=adversaries.trivial_attack
  
        train_error_list, classification_error_list=adversarial_training.classic_adversarial_training(model, epochs, train_loader, adversary, optimizer, loss_function,1,device=device)
        variables_dict={"epochs":epochs,"epsilon":0,"loss_function":loss_function,"trial_index":trl,"training_error":train_error_list,"classification_error":classification_error_list,"random_seed": seed}
        
        
        # Parent Directory path
        parent_parent_dir="models_final"
        parent_dir = "smallest_adversarial_example"
        # Directory
        trial_name="trial"+str(trl)
        if not os.path.isdir(parent_parent_dir):
            os.mkdir(parent_parent_dir)
        path = os.path.join(parent_parent_dir,parent_dir)
        if not os.path.isdir(path):
            os.mkdir(path)
        prefix=os.path.join(path,trial_name)
        if not os.path.isdir(prefix):
            os.mkdir(prefix)
        prefix=os.path.join(prefix,trial_name)

  

  
        
 
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
