import torch
import os
from train_networks import Net_sigmoid
import binary_torch_svm
import ntk_methods
import numpy as np
import adversaries
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import torch.nn.functional as F
import platform

#eps_list: list of perturbation radiuses at which the models were trained
#nets: list of lists of neural nets, nets[i] is a list of neural nets adversarially trained with perturbation radius=eps_list[i]----assumes nets[i] and svms[i] are the same length for each i
#svms: list of lists binary_torch_SVMs, svms[i] is a list of binary_torch_SVMs trained with the NTK corresponding to the neural net with perturbation radius=eps_list[i]----assumes nets[i] and svms[i] are the same length for each i
#test_set: the points to be attacked, either as a Dataset object or as a list of (x,label) pairs 
#input_nn_attack: the attack function for attacking a neural net. This function takes (model,x,label,epsilon), as inputs and outputs a tensor with the same dimensions as x as the adversarial example
#input_svm_attack: the attack function for attacking a binary_torch_SVM. This function takes (model,x,label,epsilon), as inputs and outputs a tensor with the same dimensions as x as the adversarial example
#save_fig_folder: if save_fig_folder is not None, the resulting figure would be saved in save_fig_folder

#for each neural net in nets[k][i] and binary_torch_SVM in svms[k][i], computes the average classification error under a white-box attack
def white_box_attacks_plots_vs_eps(eps_list,nets,svms,test_set,input_nn_attack,input_svm_attack,save_fig_folder=None):
    net_errors=[0]*len(eps_list)
    net_sd=[0]*len(eps_list)
    svm_errors=[0]*len(eps_list)
    svm_sd=[0]*len(eps_list)
    for i in range(len(eps_list)):
        eps=eps_list[i]
        if eps==0:
            nn_attack=adversaries.trivial_attack
            svm_attack=adversaries.trivial_attack
        else:
            def nn_attack(model,inputs,targets):
                return input_nn_attack(model,inputs,targets,eps)
            def svm_attack(model,inputs, targets):
                return input_svm_attack(model,inputs,targets,eps)
        nets_list=nets[i]
        svms_list=svms[i]

        #errors and standard deviations of a neural net
        nn_errors_eps=[classification_error_under_attack(model,model,nn_attack,test_set,True) for model in nets_list]
        nn_errors_eps=np.asarray(nn_errors_eps)
        net_errors[i]=np.mean(nn_errors_eps)
        net_sd[i]=np.std(nn_errors_eps)

        #errors and standard deviations of the svm
        svm_errors_eps=[classification_error_under_attack(model,model,svm_attack,test_set,False) for model in svms_list]
        svm_errors_eps=np.asarray(svm_errors_eps)
        svm_errors[i]=np.mean(svm_errors_eps)
        svm_sd[i]=np.std(svm_errors_eps)
       



    #plot errors
    plt.plot(eps_list,net_errors,color=(0,.5,0),label="neural net") #plot mean error of nets
    plt.scatter(eps_list,net_errors,color=(0,.5,0),label="_neural_net_dots")
    nn_upper_sd=[net_errors[i]+2*net_sd[i] for i in range(len(eps_list))]
    nn_lower_sd=[net_errors[i]-2*net_sd[i] for i in range(len(eps_list))]
    plt.plot(eps_list, nn_upper_sd, label="_nn_upper_sd",linestyle="--",color=(0,.5,0))
    plt.plot(eps_list, nn_lower_sd, label="_nn_lower_sd",linestyle="--",color=(0,.5,0))

    plt.plot(eps_list,svm_errors,color=(0,0,.5),label="SVM") #plot mean error of nets
    plt.scatter(eps_list,svm_errors,color=(0,0,.5),label="_SVM_dots")
    svm_upper_sd=[svm_errors[i]+2*svm_sd[i] for i in range(len(eps_list))]
    svm_lower_sd=[svm_errors[i]-2*svm_sd[i] for i in range(len(eps_list))]
    plt.plot(eps_list, svm_upper_sd, label="_svm_upper_sd",linestyle="--",color=(0,0,.5))
    plt.plot(eps_list, svm_lower_sd, label="_svm_lower_sd",linestyle="--",color=(0,0,.5))

    plt.xlabel("perturbation radius")
    plt.ylabel("classification error")
    plt.title("Error Under Attack")
    plt.legend()

    plt.savefig(os.path.join(save_fig_folder,"white_box"),bbox_inches='tight')
    plt.close()



#eps_list: list of perturbation radiuses at which the models were trained
#nets: list of lists of neural nets, nets[i] is a list of neural nets adversarially trained with perturbation radius=eps_list[i]----assumes nets[i] and svms[i] are the same length for each i
#svms: list of lists binary_torch_SVMs, svms[i] is a list of binary_torch_SVMs trained with the NTK corresponding to the neural net with perturbation radius=eps_list[i]----assumes nets[i] and svms[i] are the same length for each i
#test_set: the points to be attacked, either as a Dataset object or as a list of (x,label) pairs 
#input_nn_attack: the attack function for attacking a neural net. This function takes (model,x,label,epsilon), as inputs and outputs a tensor with the same dimensions as x as the adversarial example
#input_svm_attack: the attack function for attacking a binary_torch_SVM. This function takes (model,x,label,epsilon), as inputs and outputs a tensor with the same dimensions as x as the adversarial example
#save_fig_folder: if save_fig_folder is not None, the resulting figure would be saved in save_fig_folder

#for each neural net in nets[k][i] computes the average classification error on adversarial examples found by attacking svms[k][i] and
#for each binary_torch_SVM in svms[k][i] computes the average classification error on adversarial examples found by attacking nets[k][i]

#assumes that svms[k][i] is the binary torch svm found by training with the NTK corresponding to nets[k][i]
def grey_box_attacks_plots_vs_eps(eps_list,nets,svms,test_set,input_nn_attack,input_svm_attack,save_fig_folder=None):
    net_errors=[0]*len(eps_list)
    net_sd=[0]*len(eps_list)
    svm_errors=[0]*len(eps_list)
    svm_sd=[0]*len(eps_list)
    for i in range(len(eps_list)):
        eps=eps_list[i]
        if eps==0:
            nn_attack=adversaries.trivial_attack
            svm_attack=adversaries.trivial_attack
        else:
            def nn_attack(model,inputs,targets):
                return input_nn_attack(model,inputs,targets,eps)
            def svm_attack(model,inputs, targets):
                return input_svm_attack(model,inputs,targets,eps)
        nets_list=nets[i]
        svms_list=svms[i]

        #errors and standard deviations of a neural net
        nn_errors_eps=[classification_error_under_attack(svms_list[i],nets_list[i],svm_attack,test_set,True) for i in range(len(nets_list))]
        nn_errors_eps=np.asarray(nn_errors_eps)
        net_errors[i]=np.mean(nn_errors_eps)
        net_sd[i]=np.std(nn_errors_eps)

        #errors and standard deviations of the svm
        svm_errors_eps=[classification_error_under_attack(nets_list[i],svms_list[i],nn_attack,test_set,False) for i in range(len(nets_list))] 
        svm_errors_eps=np.asarray(svm_errors_eps)
        svm_errors[i]=np.mean(svm_errors_eps)
        svm_sd[i]=np.std(svm_errors_eps)
       



       #plot errors
    plt.plot(eps_list,net_errors,color=(0,.5,0),label="SVM-to-neural net") #plot mean error of nets
    plt.scatter(eps_list,net_errors,color=(0,.5,0),label="_neural_net_dots")
    nn_upper_sd=[net_errors[i]+2*net_sd[i] for i in range(len(eps_list))]
    nn_lower_sd=[net_errors[i]-2*net_sd[i] for i in range(len(eps_list))]
    plt.plot(eps_list, nn_upper_sd, label="_nn_upper_sd",linestyle="--",color=(0,.5,0))
    plt.plot(eps_list, nn_lower_sd, label="_nn_lower_sd",linestyle="--",color=(0,.5,0))

    plt.plot(eps_list,svm_errors,color=(0,0,.5),label="neural net-to-SVM") #plot mean error of nets
    plt.scatter(eps_list,svm_errors,color=(0,0,.5),label="_SVM_dots")
    svm_upper_sd=[svm_errors[i]+2*svm_sd[i] for i in range(len(eps_list))]
    svm_lower_sd=[svm_errors[i]-2*svm_sd[i] for i in range(len(eps_list))]
    plt.plot(eps_list, svm_upper_sd, label="_svm_upper_sd",linestyle="--",color=(0,0,.5))
    plt.plot(eps_list, svm_lower_sd, label="_svm_lower_sd",linestyle="--",color=(0,0,.5))

    plt.xlabel("perturbation radius")
    plt.ylabel("classification error")
    plt.title("Error Under Transfer Attack of Neural Nets and Associated SVMs")
    plt.legend()

    plt.savefig(os.path.join(save_fig_folder,"grey_box"),bbox_inches='tight')
    plt.close()



#eps_list: list of perturbation radiuses at which the models were trained
#nets: list of lists of neural nets, nets[i] is a list of neural nets adversarially trained with perturbation radius=eps_list[i]----assumes nets[i] and svms[i] are the same length for each i
#svms: list of lists binary_torch_SVMs, svms[i] is a list of binary_torch_SVMs trained with the NTK corresponding to the neural net with perturbation radius=eps_list[i]----assumes nets[i] and svms[i] are the same length for each i
#test_set: the points to be attacked, either as a Dataset object or as a list of (x,label) pairs 
#input_nn_attack: the attack function for attacking a neural net. This function takes (model,x,label,epsilon), as inputs and outputs a tensor with the same dimensions as x as the adversarial example
#input_svm_attack: the attack function for attacking a binary_torch_SVM. This function takes (model,x,label,epsilon), as inputs and outputs a tensor with the same dimensions as x as the adversarial example
#save_fig_folder: if save_fig_folder is not None, the resulting figure would be saved in save_fig_folder


#for each perturbation radius in eps_list, computes the average classification error of black box attacks computed by attacking different models.
#for example, nets[k][i] would be attacked by nets[k][j] and svms[k][j] for j!=i
def black_box_attacks_plots_vs_eps(eps_list,nets,svms,test_set,input_nn_attack,input_svm_attack,save_fig_folder=None):
    net_to_net_errors=[0]*len(eps_list)
    net_to_net_sd=[0]*len(eps_list)
    svm_to_net_errors=[0]*len(eps_list)
    svm_to_net_sd=[0]*len(eps_list)
    net_to_svm_errors=[0]*len(eps_list)
    net_to_svm_sd=[0]*len(eps_list)
    svm_to_svm_errors=[0]*len(eps_list)
    svm_to_svm_sd=[0]*len(eps_list)
    for i in range(len(eps_list)):
        eps=eps_list[i]
        if eps==0:
            nn_attack=adversaries.trivial_attack
            svm_attack=adversaries.trivial_attack
        else:
            def nn_attack(model,inputs,targets):
                return input_nn_attack(model,inputs,targets,eps)
            def svm_attack(model,inputs, targets):
                return input_svm_attack(model,inputs,targets,eps)
        #errors and standard deviations of a net_vs_net
        net_to_net_errors[i],net_to_net_sd[i]=average_non_matching_attack_mean_and_stds(nets[i],nets[i],test_set,nn_attack,True)
        svm_to_net_errors[i],svm_to_net_sd[i]=average_non_matching_attack_mean_and_stds(svms[i],nets[i],test_set,svm_attack,True)
        net_to_svm_errors[i],net_to_svm_sd[i]=average_non_matching_attack_mean_and_stds(nets[i],svms[i],test_set,nn_attack,False)
        svm_to_svm_errors[i],svm_to_svm_sd[i]=average_non_matching_attack_mean_and_stds(svms[i],svms[i],test_set,svm_attack,False)
       



       #plot errors
    net_to_net_color=(0,.5,0)
    svm_to_net_color=(.85,0,0)
    net_to_svm_color=(0,0,.85)
    svm_to_svm_color=(0,0,0)
    net_to_net_label="neural net-to-neural net"
    svm_to_net_label="SVM-to-neural net"
    net_to_svm_label="neural net-to-SVM"
    svm_to_svm_label="SVM-to-SVM"
    plot_non_matching_error(eps_list,net_to_net_errors,net_to_net_sd,net_to_net_color,net_to_net_label)
    plot_non_matching_error(eps_list,svm_to_net_errors,svm_to_net_sd,svm_to_net_color,svm_to_net_label)
    plot_non_matching_error(eps_list,net_to_svm_errors,net_to_svm_sd,net_to_svm_color,net_to_svm_label)
    plot_non_matching_error(eps_list,svm_to_svm_errors,svm_to_svm_sd,svm_to_svm_color,svm_to_svm_label)

    plt.xlabel("perturbation radius")
    plt.ylabel("classification error")
    plt.title("Error Under Transfer Attacks")
    plt.legend()

    plt.savefig(os.path.join(save_fig_folder,"black_box"),bbox_inches='tight')
    plt.close()

#helper function for plotting mean error with standard deviations
#eps_list: list of perturbation radiuses to plot on x-axis
#means: list of mean error at each perturbation radius
#stds: list of standard deviations at each perturbation radius
#color: color for plotting lines
def plot_non_matching_error(eps_list,means,stds,color,label):
    plt.plot(eps_list,means,color=color,label=label )#plot mean error of nets
    plt.scatter(eps_list,means,color=color,label="_"+label+"_dots")
    nn_upper_sd=[means[i]+2*stds[i] for i in range(len(eps_list))]
    nn_lower_sd=[means[i]-2*stds[i] for i in range(len(eps_list))]
    plt.plot(eps_list, nn_upper_sd, label="_"+label+"_upper_sd",linestyle="--",color=color)
    plt.plot(eps_list, nn_lower_sd, label="_"+label+"nn_lower_sd",linestyle="--",color=color)


#assumes attack_models and test_models are the same length,
#computes the average error of attacking test_models[i] with adversarial examples from attack_models[j] for i !=j
#attack_models: list of models to be attacked. Assumes that these are either all neural nets or all binary_torch_SVMs
#test_models: the models on which to evaluate the adversarial examples. Assumes that these are either all neural nets with two outputs or all binary_torch_SVMs
#test_set: the points to be attacked, either as a Dataset object or as a list of (x,label) pairs 
#attack: the attack function. This function takes (model,x,label), as inputs and outputs a tensor with the same dimensions as x as the adversarial example
#test_model_neural_net: set to true if test_models is a list of neural nets and false if its a list of binary_torch_SVMs
def average_non_matching_attack_mean_and_stds(attack_models,test_models,test_set,attack,test_model_neural_net):
    M=len(attack_models)
    mns=[]
    for test_index in range(M):
        for attack_index in range(M):
            if test_index != attack_index:
                error=classification_error_under_attack(attack_models[attack_index],test_models[test_index],attack,test_set,test_model_neural_net)
                mns.append(error)
    mns=np.asarray(mns)
    mean_error=np.mean(mns)
    sd=np.std(mns)
    return mean_error,sd
    


#loads pre-trained models, extracts the associated ntks, and trains the associated svms. Returns a list of models and a list of 
#eps: the radius of perturbation used in adversarial training
#train_set: the data set on which to compute the NTK. can be a data set object or a list of (x,label) pairs
def load_models(eps,train_set):
    Y=[label for (x,label) in train_set]

    trial_name_base="trial"
    parent_parent_dir="models_final"
    parent_dir = "binary_sigmoid_models"
    models_list=[]
    svm_list=[]
    for i in range(0,10):
        trial_name="'"+trial_name_base+str(i)+"'"
        # Directory

        path = os.path.join(parent_parent_dir,parent_dir)
        prefix=os.path.join(path,trial_name)
        prefix=os.path.join(prefix,trial_name)
        model_name=prefix+"_epsilon="+str(eps)+"model.pt"
        net=Net_sigmoid()
        state_dict=torch.load(model_name,map_location=torch.device('cpu'))
        net.load_state_dict(state_dict)
        net.eval()
        models_list.append(net)
        feature_mapping=ntk_methods.feature_mapping(net, train_set, data_set=True, higher_derivatives=False)
        svm=ntk_methods.svm_from_kernel_matrix(feature_mapping,Y)
        torch_svm=binary_torch_svm.binary_torch_SVM(net,svm,feature_mapping)
        svm_list.append(torch_svm)
    
    return models_list,svm_list



  


#computes the classification error of a model under attack. The model under attack is test_model, and adversarial perturbations are found by attacking the attack_model
#attack_model: the model to be attacked. Assumes that this model is either a neural net or a binary_torch_SVM
#test_model: the model on which to evaluate the adversarial examples. Assumes that this model is either a neural net with two outputs or a binary_torch_SVM
#attack: the attack function. This function takes (model,x,label), as inputs and outputs a tensor with the same dimensions as x as the adversarial example
# dataset: the points to be attacked, either as a Dataset object or as a list of (x,label) pairs 
def classification_error_under_attack(attack_model,test_model,attack,dataset,test_model_neural_net):
    total=0
    correct=0

    for x,label in dataset:
        total=total+1
        torch_label=torch.tensor([label])
        #torch_label=torch_label[None,:]
        adversarial_example=attack(attack_model,x,torch_label)
        if predict(test_model,adversarial_example,test_model_neural_net)==label:
            correct=correct+1
    return 1-correct/total





#function for predicting the output of a model at a point x
#model:either a neural net or a binary_torch_SVM
#x: a tensor at which we want the model prediction
#neural_net: true if model is a neural net, false if it is a binary_torch_SVM
def predict(model, x, neural_net):
    y=model(x)[0]
    if neural_net:
        if y[0].item()>y[1].item():
            return 0
        else:
            return 1
    else:
        if y.item()<0:
            return 0
        else:
            return 1


#the identity function; to assist in attacking a binary_torch_SVM
def identity_loss_function(x,label):
    return x



def main():

    mnist_train_set= datasets.MNIST('..\data', train=True, download=True, transform=transforms.Compose([
        transforms.ToTensor(),]))

    #the models were trained on the subset of MNIST comprised of just 1s and 7s. We extract this subset of the training set
    mnist_train_sbset=[]
    for (data, label) in mnist_train_set:
        data=data[None,:]
        if label==1:
            mnist_train_sbset.append((data,label))
        elif label==7:
            mnist_train_sbset.append((data,0))
    #We extract the 1s and 7s of the test set
    mnist_test_set= datasets.MNIST('C:\\Users\\fran316\\OneDrive - PNNL\\Documents\\data2', train=True, download=True, transform=transforms.Compose([
        transforms.ToTensor(),]))
    mnist_test_sbset=[]
    for (data, label) in mnist_test_set:
        data=data[None,:]
        if label==1:
            mnist_test_sbset.append((data,label))
        elif label==7:
            mnist_test_sbset.append((data,0))
    

#uncomment these lines to test the code on a small portion of MNIST
    #mnist_train_sbset=mnist_train_sbset[0:10]
    #mnist_test_sbset=mnist_test_sbset[0:2]


    #list of perturbation radiuses for which the models were trained
    eps_list=[0,0.05,0.1,0.15,0.2,0.25, 0.3]

    nets=[]
    svms=[]
    
    #loading all the models and training the SVM
    for eps in eps_list:
        nets_list,svm_list=load_models(eps,mnist_train_sbset)
        nets.append(nets_list)
        svms.append(svm_list)

    loss_function=F.cross_entropy
    p=float('inf')
    iter=7
    device='cpu'
    #models were trained with this attack at various levels of epsilon
    def input_nn_attack(model,x,y,eps):
        return adversaries.pgd_attack_p(x,y,eps,model,loss_function,p,iter,device=device,rand_init=True)

    #same attack on the svm
    def input_svm_attack(model,x,y,eps):
        return adversaries.pgd_attack_p(x,y,eps,model,identity_loss_function,p,iter,device=device,rand_init=True)
    
    #create folder for plots if it doesn't already exist
    plots_parent_dir=os.path.join("models_final","binary_sigmoid_models","plots")
    if not os.path.exists(plots_parent_dir):
        os.mkdir(plots_parent_dir)
    
    #make plots for each of the attacks
    white_box_attacks_plots_vs_eps(eps_list,nets,svms,mnist_test_sbset,input_nn_attack,input_svm_attack,save_fig_folder=plots_parent_dir)
    grey_box_attacks_plots_vs_eps(eps_list,nets,svms,mnist_test_sbset,input_nn_attack,input_svm_attack,save_fig_folder=plots_parent_dir)
    black_box_attacks_plots_vs_eps(eps_list,nets,svms,mnist_test_sbset,input_nn_attack,input_svm_attack,save_fig_folder=plots_parent_dir)

    




if __name__ == "__main__":
    if platform.platform() == "Windows-10-10.0.19042-SP0":
        os.chdir("C:\\Users\\fran316\\OneDrive - PNNL\\Documents\\experiments")
    else:
        os.chdir("/people/fran316/experiments")
    main()