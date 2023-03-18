
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
from train_full_MNIST_networks import Net_relu
import copy
import ntk_methods
import time







def main():
    trial_num=10
    test_set= datasets.MNIST('../pretrained_data', train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            ]))


    train_set= datasets.MNIST('../pretrained_data', train=True, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
           ]))

    #uncomment these lines to test this method on a small dataset
    #sbset=list(range(0,10))
    #test_set=torch.utils.data.Subset(test_set,sbset)
    #sbset=list(range(0,11))
    #train_set=torch.utils.data.Subset(train_set,sbset)
    #trial_num=2
    
    train_data_fraction=1/2
    data_num=0 #the size of the dataset
    for d in train_set:
        data_num+=1
    
    test_set=torch.utils.data.DataLoader(test_set,batch_size=1)

    loss_function=F.cross_entropy
    iterations=20
    repetitions=10
    p=float("inf")
    #it's hard to find the optimal adversarial example, this is a very strong adversary which can be used as a proxy.
    def pgd_adversary(x,label,epsilon,model):
        return adversaries.pgd_repetitions(x,label,epsilon,model,loss_function,p,iterations,repetitions=repetitions,include_center_initialization=True)

    #note that the adversary defined about includes randomness
    #setting the seed:
    seed= int(time.time())
    torch.manual_seed(seed)

    #setting up directories
    trial_base_name='trial'
    parent_parent_dir="models_final"
    parent_dir = "smallest_adversarial_example"
    pth = os.path.join(parent_parent_dir,parent_dir)
    plots_dir=os.path.join(pth,"plots")
    pickles_dir=os.path.join(pth,"pickles")
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    if not os.path.exists(pickles_dir):
        os.makedirs(pickles_dir)

    #saving the seed:
    seed_path=os.path.join(pickles_dir, "random_seed")
    file=open(seed_path,'wb')
    pickle.dump(seed,file)

    cos_sims_strong_pgd_list=[]
    cos_sims_trivial_list=[]
    for i in range(trial_num):
        print(str(i))

        trial_name=trial_base_name+str(i)

        # Parent Directory path
        # Directory
        trial_name=trial_base_name+str(i)
        
        prefix=os.path.join(pth,trial_name)
        prefix=os.path.join(prefix,trial_name)

        #load the neural net
        model_name=prefix+"model.pt"
        model=Net_relu()
        model.load_state_dict(torch.load(model_name, map_location='cpu'))
        model.eval()

        #we evaluate on a fraction of the training data to avoid running out of memory
        subset_size=round(data_num*train_data_fraction)
        perm=np.random.permutation(data_num)
        sbset=list(perm[0:subset_size])
        train_sbset=torch.utils.data.Subset(train_set,sbset)


        #computing the feature mapping for the ntk
        feature_mapping=ntk_methods.feature_mapping(model,train_sbset,return_torch_tensor=False)

        #calculating and saving the cosine similarities between the smallest adversarial attack and the feature mapping
        cos_sims_strong_pgd_list.append(smallest_adversarial_example.cosine_similarities_at_smallest_perturbation_vectorized(test_loader=test_set,model=model,
                feature_mapping=feature_mapping,attack=pgd_adversary,upper_bound_guess=1,strict=True))

        #calculating and saving the cosine similarities between natural data and the feature mapping
        lsts=smallest_adversarial_example.cosine_similarities_of_attack_vectorized(test_loader=test_set,model=model,
            feature_mapping=feature_mapping,attack=adversaries.trivial_attack)
        lst=np.concatenate((lsts[0],lsts[1]))
        cos_sims_trivial_list.append(lst)
    


    #saving the cosine similarities for the smallest adversarial perturbations
    file_name_base_pgd='full_mnist_10_trials_strong_pgd'
    file_name_pgd_pickle=os.path.join(pickles_dir,file_name_base_pgd)
    file=open(file_name_pgd_pickle,'wb')
    pickle.dump(cos_sims_strong_pgd_list,file)

    #plotting the histogram of cosine similarities for smallest adversarial perturbations
    figure_pth=os.path.join(plots_dir,file_name_base_pgd)
    smallest_adversarial_example.attribute_histogram(cos_sims_strong_pgd_list,save_file=figure_pth,rng=(-1,1),bins=None,x_label="cosine similarity", title="Cosine Similarity for Smallest Adversarial Examples")

    #saving the cosine similarities for natural data
    file_name_base_trivial='full_mnist_10_trials_no_adversary'
    file_name_cossim_trivial_pickle=os.path.join(pickles_dir,file_name_base_trivial+'_cossim.p')
    file=open(file_name_cossim_trivial_pickle,'wb')
    pickle.dump(cos_sims_trivial_list,file)
    

    #plotting the histogram of cosine similarities for natural data
    figure_pth=os.path.join(plots_dir,file_name_base_trivial)
    smallest_adversarial_example.attribute_histogram(cos_sims_trivial_list,save_file=figure_pth,rng=(-1,1),bins=None,x_label="cosine similarity",title="Cosine Similarity with NTK for Natural Data")




if __name__ == "__main__":
    if platform.platform() == "Windows-10-10.0.19042-SP0":
        os.chdir("C:\\Users\\fran316\\OneDrive - PNNL\\Documents\\experiments")
    else:
        os.chdir("/people/fran316/experiments")

    main()