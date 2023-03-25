from .base import BaseModel, Setting, OUTPUT_ACTIVATIONS
import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeSeriesMultiCNN(BaseModel):
    """ Model implementing multi-layer time series CNN """
    def __init__(
            self, in_features, out_features, kernel_size=3, hidden_units=64,
            groups=1, causal=True, num_layers=3, out_activation=None
            ):
        super().__init__(in_features, out_features)
        self.kernel_size = Setting(kernel_size)
        self.hidden_units = Setting(hidden_units)
        self.causal = Setting(causal)
        self.groups = Setting(groups)
        self.num_layers = Setting(num_layers)
        self.out_activation = Setting(out_activation)
        self.configure()
    
    @classmethod
    def from_checkpoint(cls, checkpoint):
        cfg = checkpoint["config"]
        out = cls(
            cfg["in_features"], cfg["out_features"], cfg["kernel_size"],
            cfg["hidden_units"], cfg["groups"], cfg["causal"], cfg["num_layers"],
            cfg.get("out_activation", None)
            )
        out.load_state_dict(checkpoint["state_dict"])
        return out
    
    @staticmethod
    def description():
        return "1-D multi-layer CNN with linear output layer"
    
    def configure(self):
        nets = [
            nn.Conv1d(
                self.in_features, self.hidden_units, self.kernel_size,
                groups=self.groups
            )
        ]
        for _ in range(1, self.num_layers):
            nets.append(
                nn.Conv1d(
                        self.hidden_units, self.hidden_units, self.kernel_size,
                        groups=self.groups
                    )
            )
        # output layer is CNN with kernel size = 1
        group_output = 1 if bool(self.out_features % self.groups) else self.groups
        nets.append(
            nn.Conv1d(
                self.hidden_units, self.out_features, 1, groups=group_output
                )
        )
        self.nn = nn.ModuleList(nets)
        self.out_fun = OUTPUT_ACTIVATIONS[self.out_activation]

    def forward(self, x):
        if self.causal:
            padl = self.kernel_size - 1
            padr = 0
        else:
            padl = (self.kernel_size - 1) // 2
            padr = (self.kernel_size) // 2
        xt = x.transpose(1,2)
        for n in self.nn[:-1]:
            xp = F.pad(xt, (padl, padr), "replicate")
            xt = F.relu(n(xp))
        
        return self.out_fun(self.nn[-1](xt).transpose(1,2))


class TimeSeriesCNN(BaseModel):
    """ Model implementing a simple time-series CNN """
    def __init__(
            self, in_features, out_features, kernel_size=3, hidden_units=64, groups=1, 
            causal=True, out_activation=None):
        super().__init__(in_features, out_features)
        self.kernel_size = Setting(kernel_size)
        self.hidden_units = Setting(hidden_units)
        self.causal = Setting(causal)
        self.groups = Setting(groups)
        self.out_activation = Setting(out_activation)
        self.configure()
    
    @classmethod
    def from_checkpoint(cls, checkpoint):
        cfg = checkpoint["config"]
        out = cls(
            cfg["in_features"], cfg["out_features"], cfg["kernel_size"],
            cfg["hidden_units"], cfg["groups"], cfg["causal"],
            cfg.get("out_activation", None)
            )
        out.load_state_dict(checkpoint["state_dict"])
        return out

    @staticmethod
    def description():
        return "1-D CNN with linear output layer"

    def configure(self):
        self.cnn = nn.Conv1d(
                self.in_features, self.hidden_units, self.kernel_size, 
                groups=self.groups
                )
        if self.groups == self.in_features and self.out_features == self.in_features:
            self.net = nn.Conv1d(
                self.hidden_units, self.out_features, 1, groups=self.groups
            )
        else:
            self.net = nn.Linear(self.hidden_units, self.out_features)
        self.out_fun = OUTPUT_ACTIVATIONS[self.out_activation]
    
    def forward(self, x):
        if self.causal:
            padl = self.kernel_size - 1
            padr = 0
        else:
            padl = (self.kernel_size - 1) // 2
            padr = (self.kernel_size) // 2
        xp = F.pad(x.transpose(1, 2), (padl, padr), "replicate")
        x = F.relu(self.cnn(xp))
        if isinstance(self.net, nn.Linear):
            return self.out_fun(self.net(x.transpose(1,2)))
        else:
            return self.out_fun(self.net(x).transpose(1,2))

