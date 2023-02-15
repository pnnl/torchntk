# TorchNTK

#### An Arbitrary** PyTorch Architecture Neural Tangent Kernel Library

### Installation

* git clone this repository

```bash
git clone https://github.com/pnnl/torchntk
```

* Add to PYTHONPATH

```bash
export PYTHONPATH="${PYTHONPATH}:/my/path/TorchNTK/"
```

* Make sure you have correct dependencies installed; Broadly, this code was tested with PyTorch 1.9, numba 0.53.1, and Tensorboard 2.6.0, on Python 3.8.8.

* The torch.vmap function is only available on nightly releases of PyTorch. torch.vmap is only used for one implementation of an autograd calculation-- *it is not required*

* For the notebooks comparing to neural tangents, you will also need jax, jaxlib, and neural-tangents installed. This can be tricky for windows users, and we suggest going to the original neural-tangents page for detailed installation instructions [here](https://github.com/google/neural-tangents)

* For the tensorboard.ipynb notebook, download the dataset from [here](https://drive.google.com/file/d/1MwD_k-KLlfins4wKiTdCmTIfUfzlKJwg/view?usp=sharing) and place into ./DATA/ ; though you very well could use any other dataset or simulated data.

### Basic Usage

```python

import torchntk
import torch

DEVICE = 'cpu' #or cuda, lets say

model = Pytorch_Model() #Any architecture-- BUT must terminate in single neuron
model.to(DEVICE)

Y = model(X) 

NTK_components = torchntk.autograd.autograd_components_ntk(model,Y)
```

or, a generally faster implementation exists if torch.vmap exists (currently available in pytorch nightly builds only)

```python

import torchntk
import torch
from torch.utils.data import DataLoader, TensorDataset

DEVICE = 'cuda' #

model = Pytorch_Model() #Any architecture-- BUT must terminate in single neuron
model.to(DEVICE)

xloader = DataLoader(TensorDataset(My_data,My_targets),batch_size=64, shuffle=False)

NTK_components = torchntk.autograd.vmap_ntk_loader(model,xloader)
```

Finally, if you are using a fully connected network (a network composed only if torch.nn.Linear layers) you can use this last method which is typically much faster:

```python
import torchntk
import torch

DEVICE = 'cuda'

def activation(X):
    return torch.tanh(X)
	
def d_activation(X):
    return torch.cosh(X)**-2

class MLP(torch.nn.Module):
    def __init__(self,):
        super(MLP, self).__init__()
        self.d1 = torch.nn.Linear(784,100,bias=True) 
        self.d2 = torch.nn.Linear(100,100,bias=True)
        self.d3 = torch.nn.Linear(100,1,bias=True) 
    def forward(self, x_0):
        x_1 = activation(self.d1(x_0)) / torch.sqrt(100)
        x_2 = activation(self.d2(x_1)) / torch.sqrt(100)
        x_3 = activation(self.d3(x_2)) / torch.sqrt(1)
        return x_3, x_2, x_1, x_0 


model = MLP()
model.to(DEVICE)

x_3, x_2, x_1, x_0 = model(X) #for some data, X

Xs = [x_0.T.detach(),
      x_1.T.detach(),
	  x_2.T.detach()]
	  
layers = [model.d1,
          model.d2,
		  model.d3]
		  
#this must match the layer's width
ds_int = [100, 100, 1]

#this must match what you divided the layer by, squared.
#i.e., if you didn't divide each layer by anything, this should be all ones.
ds_float = [100.0, 100.0, 1.0]


config = {'Xs':Xs,
          'layers':layers,
		  'ds_int':ds_int,
		  'ds_float':ds_float,
		  'dactivation_t':d_activation}
 
components = torchntk.explicit.explicit_ntk(**config)
#components is a list of torch.Tensor objects representing each component of
#the NTK from each parameterized operation in reverse order. Meaning, 
#components[0] is the outermost layer weight matrix NTK component, 
#components[1] is the outermost layer bias vector NTK component,
# ...
#components[-1] is the first layer's bias vector NTK components 

#to get the full NTK, simply sum the components across the list's dimension.

```

### Logging with Tensorboard

Tutorial coming soon. In the meantime, you can check the tensorboard.ipynb notebook.

Once installed, Tensorboard can be started on the command line with:
```bash
tensorboard --logdir=LOGDIR
```

### Possible Metrics of Interest

The condition number is the (minimum eigenvalue of the NTK / maximum eigenvalue of the NTK). It is negatively correlated with model performance

### Credit

"torchntk.autograd.old_autograd_ntk" was directly adatapted from the TENAS group's code, available [here](https://github.com/VITA-Group/TENAS) , and you can view their paper on neural architecture seach [here](https://arxiv.org/pdf/2102.11535.pdf); authored by Chen, Wuyang and Gong, Xinyu and Wang, Zhangyang and titled: "Neural Architecture Search on ImageNet in Four GPU Hours: A Theoretically Inspired Perspective"

Some backward propogation functions were originally copied then heavily modified from this article by Pierre Jaumier, available [here](https://towardsdatascience.com/backpropagation-in-a-convolutional-layer-24c8d64d8509)

I've also included some utility functions that I directly copied from the PyTorch source; therefore, their license clause is included in ours.

Experimental autograd operations were adapted from web pages in the pre-release of Pytorch1.11; but now with Pytorch 1.11 release I advise you take a look at functorch's [NTK page](https://pytorch.org/functorch/stable/notebooks/neural_tangent_kernels.html). 

### Software TODO (or how you can contribute)

* Add explicit calculations for more varied architectures
* Parallelize computation across multiple GPUs
* make that notebook that demonstrates the different algorithms into a test such that pytest ccould be run on it, assert all outputs are ~same

