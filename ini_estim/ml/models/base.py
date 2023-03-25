import abc
import types
import torch
import torch.nn as nn
import json
import warnings


class Setting:
    __slots__ = "val"
    def __init__(self, val):
        self.val = val


OUTPUT_ACTIVATIONS = {
    "": torch.nn.Identity(),
    False: torch.nn.Identity(),
    "none": torch.nn.Identity(),
    None: torch.nn.Identity(),
    "relu": torch.relu,
    "sigmoid": torch.sigmoid,
    "tanh": torch.tanh
}



class BaseModel(nn.Module, metaclass=abc.ABCMeta):
    """ Base class for high level neural network models 
    
    BaseModel inherits from PyTorch's Module and adds extra functionality to
    make saving/restoring/training more convenient. See the MLP model for
    example usage.

    Settings:
    Any attribute defined as a Setting object will automatically be registered
    as a configuration parameter. Setting is a thin wrapper for any value and
    simply allows them to be automatically discovered. These parameters are 
    saved in the model checkpoint so that the model can later be restored with 
    the same configuration. BaseModel defines settings "in_features" and 
    "out_features", which are intended to be sizes of the last dimension of the 
    input and output. Classes inheriting from BaseModel that are intended to be
    used as "out = model(in)" should define in_features and out_features. By
    default, in_features and out_features are set to None and more complex
    models can simply ignore these settings.

    Example Usage:
        class MyModel(BaseModel):
            def __init__(self, in_features, out_features, hidden_units):
                super().__init__()
                self.in_features = in_features
                self.out_features = out_features
                self.hidden_units = Setting(hidden_units)
                self.configure()
            
            def configure(self):
                self.net1 = nn.Linear(self.in_features, self.hidden_units)
                self.net2 = nn.Linear(self.hidden_units, self.out_features)
            
            def forward(self, x):
                y = torch.relu(self.net1(x))
                return self.net2(y)
    """
    def __init__(self, in_features=None, out_features=None):
        """ Initialize all the Settings for the model. 

        Classes inheriting from BaseModel should call this first. After all
        settings are defined, the configure() method should be called to
        actually set up and define the network modules.
        """
        self._cfg = dict()
        super().__init__()
        self.in_features = Setting(in_features)
        self.out_features = Setting(out_features)

    def __getattr__(self, name):
        cfg = self.__dict__.get("_cfg")
        if name in cfg:
            return cfg[name]
        return super().__getattr__(name)

    def __setattr__(self, name, value):
        cfg = self.__dict__.get("_cfg")
        if isinstance(value, Setting):    
            if cfg is None:
                raise AttributeError("Cannot assign settings before BaseModel.__init__ is called.")
            cfg[name] = value.val
        elif cfg is not None and name in cfg:
            cfg[name] = value
        else:
            return super().__setattr__(name, value)

    @abc.abstractmethod
    def configure(self):
        """Configure the model according to the defined Settings. 

        This is where all network definitions belong.
        """
        raise NotImplementedError

    def get_config(self):
        return self._cfg.copy()

    def save_config(self, filepath):
        with open(filepath, 'w+') as f:
            json.dump(self._cfg, f, indent=2)

    def to_dict(self):
        return {
            "name": self._get_name(),
            "config": self._cfg,
            "state_dict": self.state_dict()
        }

    def save_checkpoint(self, filepath, metadata=None):
        checkpoint = self.to_dict()
        checkpoint["meta"] = metadata
        torch.save(checkpoint, filepath)
    
    def load_checkpoint(self, checkpoint):
        device = next(self.parameters()).device
        if not isinstance(checkpoint, dict):
            checkpoint = torch.load(checkpoint)
        if checkpoint["name"] != self._get_name():
            warnings.warn(
                "Checkpoint model name \"{}\" does not match \"{}\"".format(
                    checkpoint["name"], self._get_name()
            ))
        self._cfg.update(checkpoint["config"])
        self.configure()
        self.to(device)
        self.load_state_dict(checkpoint["state_dict"])
        return checkpoint.get("meta")
    
    @classmethod
    def from_checkpoint(cls, checkpoint):
        cfg = checkpoint["config"]
        out = cls(**cfg)
        out.load_state_dict(checkpoint["state_dict"])
        return out

    @classmethod
    def subclasses(cls, add_module=False):
        out = {}
        for c in cls.__subclasses__():
            module = c.__module__.split(".")[-1]
            name = c.__qualname__
            if len(module) and add_module:
                name = module + "." + name
            out[name] = c
        return out
