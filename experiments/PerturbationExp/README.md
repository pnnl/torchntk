files starting with zt are important test files
files starting with zb are sbatch files I used for the final product
all files without a prefix are important
models_final contain all the trained models for the two experiments

The two final plotting files are 
smallest_adversarial_example_plotting.py 
nn_vs_svm_attack_plotting.py

Files that should be included in the published repository are:
-files with prefix zb
-all files with no prefix EXCEPT pretrained_models//lenet_mnist_model.pth
(I apologize for this annoying exception, I've referenced this file in several tests and zz-/zz_ files. I don't want to fix all the tests, so there is an expection to the naming convention)

Inefficiencies in the code:
These are described in the associated pdf file
1. smallest_adversarial_perturbation: The QR factorization takes up 2X too much memory. Currently, I've done the calculation for a random half for the training set for each neural net. possible improvements: 1) If we could convince numpy.linalg to do computations with float32 instead of float64, this should solve the memory problem. I don't know how to do this though. 2) We should consider running multiple calculations for each neural net. However, because we are using a large portion of the training set, this source of randomness may not be the largest source of variance.
	Keep in mind: the 'vectorized' versions of functions in this file are MUCH MUCH faster
2. nn_vs_svm_attack: the most expensive part is generating the adversarial examples for the SVM. As the code is currently written, this step is done 11 times for each SVM. (adversarial examples for the neural net are not nearly as expensive) I made this choice because a)I was concerned about memory use b) I was not aware attacking the SVM would be so expensive c) it was programmatically easier