import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from ..models import Setting
from .base import BaseTrainer, BaseModel
from ..models import TimeSeriesCNN
from ..utilities import random_sample_sequence


_log2 = math.log(2)


class CPCLoss(BaseModel):
    def __init__(self, num_features, num_pred, use_jsd=False, context_network=None):
        super().__init__()
        self.num_features = Setting(num_features)
        self.num_pred = Setting(num_pred)
        self.use_jsd = Setting(use_jsd)
        self.context_network = Setting(context_network)
        self._measure = self._nce
        self.configure()
    
    def configure(self):
        self.discriminators = nn.ModuleList([
            nn.Bilinear(self.num_features, self.num_features, 1, bias=False)
            for _ in range(self.num_pred)
        ])
        self._measure = self._jsd if self.use_jsd else self._nce
        cn = "none" if self.context_network is None else self.context_network.lower()
        if cn == "none":
            self.confun = None
        elif cn == "cnn":
            print("Configuring context network...")
            self.confun = TimeSeriesCNN(
                self.num_features, self.num_features, 4*self.num_pred + 1
                )
        else:
            raise NotImplementedError(
                    "Context network type: {} is not implemented.".format(
                    self.context_network)
                )
        
    
    def forward(self, z, lengths=None):
        num_batch, max_length, num_dims = z.shape
        winsize = max_length - self.num_pred - 1 
        if lengths is not None:
            lengths = lengths - self.num_pred - 1
        num_neg = num_batch
        
        # run context process
        c0 = z[:, :winsize, :].contiguous()
        if self.confun is not None:
            c0 = self.confun(c0)
        
        zn = random_sample_sequence(z, num_neg, lengths, winsize)
        c0multi = c0.expand_as(zn).contiguous()

        loss = 0.0
        for k in range(1, self.num_pred+1):
            zk = z[:, k:winsize+k, :].contiguous()
            p = self.discriminators[k-1](c0, zk)
            q = self.discriminators[k-1](c0multi, zn)
            if lengths is not None:
                for i, l in enumerate(lengths):
                    p[i, l:] = 0
                    q[i, :, l:] = 0
            loss += self._measure(p, q, lengths)
        return loss / self.num_pred
    
    def _nce(self, p, q, lengths=None):
        en = torch.sum(torch.logsumexp(torch.cat((p.unsqueeze(1), q), 1), 1))
        ep = torch.sum(p)
        N = p.numel() if lengths is None else torch.sum(lengths)
        return (en - ep) / max(N, 1)

    def _jsd(self, p, q, lengths=None):
        Np = p.numel() if lengths is None else torch.sum(lengths)
        Nq = q.numel() if lengths is None else torch.sum(lengths)*q.shape[1]
        ep = torch.sum(_log2 - F.softplus(-p)) / max(Np, 1)
        en = torch.sum(F.softplus(-q) + q - _log2) / max(Nq, 1)
        return en - ep


class CPCTrainer(BaseTrainer):
    def __init__(
            self, model, num_pred=10, use_jsd=False, context_net=None, 
            model_lr=1e-3, loss_lr=1e-3, save_dir=None, save_interval=5,
            **kwargs
            ):
        super().__init__(save_dir=save_dir, save_interval=save_interval, **kwargs)
        self.model = model
        self.model.to(self.device)
        self.loss = CPCLoss(
            self.model.out_features, num_pred, use_jsd, 
            context_network=context_net
            )
        self.loss.to(self.device)
        self.model_lr = model_lr
        self.loss_lr = loss_lr
        self.apply_optimizers()
    
    def apply_optimizers(self):
        self.optimizer = [
            optim.Adam(self.model.parameters(), lr=self.model_lr),
            optim.Adam(self.loss.parameters(), lr=self.loss_lr)
        ]

    def get_loss(self, batch):
        if len(batch) == 3:
            data, lengths, _ = batch
        else:
            data, _ = batch
            lengths = None
        data = data.to(self.device)
        encoded = self.model(data)
        return self.loss(encoded, lengths)

    def train_step(self, batch):
        loss = self.get_loss(batch)
        loss.backward()
        for o in self.optimizer:
            o.step()
        for o in self.optimizer:
            o.zero_grad()
        return loss.item()
    
    def eval_step(self, batch):
        return self.get_loss(batch).item()
    
