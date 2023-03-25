from .base import BaseModel, Setting, OUTPUT_ACTIVATIONS
import torch.nn as nn
import torch.nn.functional as F


class MLP(BaseModel):
    """ Model implementing a multi-layer perceptron  """
    def __init__(
            self, layer_sizes, activation="relu", out_activation=None, 
            flatten=True
        ):
        """
        Parameters
        ----------
        layer_sizes : list of ints
            list of layer dimensions to use
        activation : str
            Module to use as activation function, "relu" by default. 
            Valid options are "relu", "tanh", "sigmoid". An empty string or
            None will omit the activation function.
        out_activation : str, optional
            Type of output activation. Set to None to disable. Supported options are
            "relu", "sigmoid", "tanh"
        flatten : bool
            Boolean whether to flatten the input first. Default is True.
        
        """    
        super().__init__(layer_sizes[0], layer_sizes[-1])
        self.layer_sizes = Setting(layer_sizes)
        self.activation = Setting(activation)
        self.out_activation = Setting(out_activation)
        self.flatten = Setting(flatten)
        self.configure()
    
    @classmethod
    def from_checkpoint(cls, checkpoint):
        cfg = checkpoint["config"]
        out = cls(
            cfg["layer_sizes"], cfg["activation"], cfg["out_activation"],
            cfg["flatten"]
            )
        out.load_state_dict(checkpoint["state_dict"])
        return out

    @staticmethod
    def description():
        return "Multi-layer perceptron (aka feed-forward neural network)."

    def configure(self):
        """ Create a MLP """
        modules = []
        activations = {
            "relu": nn.ReLU,
            "tanh": nn.Tanh,
            "sigmoid": nn.Sigmoid,
            "": None,
            None: None
        }
        self.in_features = self.layer_sizes[0]
        self.out_features = self.layer_sizes[-1]
        f_act = activations[self.activation]
        if self.flatten:
            modules.append(nn.Flatten())
        for i, l in enumerate(self.layer_sizes[:-2]):
            modules.append(
                nn.Linear(l, self.layer_sizes[i+1])
            )
            if f_act is not None:
                modules.append(f_act())
        modules.append(
            nn.Linear(self.layer_sizes[-2], self.layer_sizes[-1])
            )
        self.net = nn.Sequential(*modules)
        self.f_out =  OUTPUT_ACTIVATIONS[self.out_activation]

    def forward(self, x):
        return self.f_out(self.net(x))

