from .base import BaseModel, Setting, OUTPUT_ACTIVATIONS
import torch
import torch.nn as nn
import torch.nn.functional as F



class Diagonal(BaseModel):
    """ Model implementing a diagonal linear transformation  """
    def __init__(
            self, features, out_activation=None
        ):
        """
        Parameters
        ----------
        features : int
            Number of input and output features
        out_activation : str, optional
            Type of output activation. Set to None to disable. Supported options are
            "relu", "sigmoid", "tanh"
        
        """    
        super().__init__(features, features)
        self.out_activation = Setting(out_activation)
        self.configure()
    
    @classmethod
    def from_checkpoint(cls, checkpoint):
        cfg = checkpoint["config"]
        out = cls(cfg["in_features"], cfg["out_activation"])
        out.load_state_dict(checkpoint["state_dict"])
        return out

    @staticmethod
    def description():
        return "Linear transformation with optional output activation"

    def configure(self):
        """ Create network """
        vals = torch.rand(self.in_features) - 0.5
        vals = vals / torch.norm(vals)
        bias = torch.rand(self.in_features) - 0.5
        bias = bias / torch.norm(bias)
        self.vals = nn.Parameter(vals)
        self.bias = nn.Parameter(bias)
        self.f_out =  OUTPUT_ACTIVATIONS[self.out_activation]

    def forward(self, x):
        x = F.linear(x, torch.diag_embed(self.vals), self.bias)
        return self.f_out(x)


class Linear(BaseModel):
    """ Model implementing a linear transformation  """
    def __init__(
            self, in_features, out_features, out_activation=None
        ):
        """
        Parameters
        ----------
        in_features : int
            Number of input features
        out_features : int
            Number of output features
        out_activation : str, optional
            Type of output activation. Set to None to disable. Supported options are
            "relu", "sigmoid", "tanh"
        
        """    
        super().__init__(in_features, out_features)
        self.out_activation = Setting(out_activation)
        self.configure()
    
    @classmethod
    def from_checkpoint(cls, checkpoint):
        cfg = checkpoint["config"]
        out = cls(
            cfg["in_features"], cfg["out_features"], cfg["out_activation"]
            )
        out.load_state_dict(checkpoint["state_dict"])
        return out

    @staticmethod
    def description():
        return "Linear transformation with optional output activation"

    def configure(self):
        """ Create network """
        self.net = nn.Linear(self.in_features, self.out_features)
        self.f_out =  OUTPUT_ACTIVATIONS[self.out_activation]

    def forward(self, x):
        return self.f_out(self.net(x))

