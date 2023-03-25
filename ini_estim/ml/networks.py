import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, layer_sizes, f_act=nn.ReLU, f_out=None, flatten=True):
        """ Create a MLP 

        Convenience class for automated creation of multi-layer perceptron.
        Uses torch.nn.Sequential.

        Parameters
        ----------
        layer_sizes : list of ints
            list of layer dimensions to use
        f_act : torch.nn.Module
            Module to use as activation function, the default is nn.ReLU
        f_out : torch.nn.Module
            Module to use as final activation function. Default is None.
        flatten : bool
            Boolean whether to flatten the input first. Default is True.
        """
        super().__init__()
        modules = []
        if flatten:
            modules.append(nn.Flatten())
        for i, l in enumerate(layer_sizes[:-2]):
            modules.append(
                nn.Linear(l, layer_sizes[i+1])
            )
            if f_act is not None:
                modules.append(f_act())
        modules.append(nn.Linear(layer_sizes[-2], layer_sizes[-1]))
        if f_out is not None:
            modules.append(f_out())
        self.fun = nn.Sequential(*modules)
    
    def forward(self, x):
        return self.fun(x)
