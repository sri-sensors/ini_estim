import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from .base import BaseTrainer
from ..models import BaseModel, Setting
from ..utilities import random_sample_sequence
from ..discriminator_models import MLPDiscriminator
from .direct_infomax import jsd_expectations


class SeriesPredictiveCodingLoss(BaseModel):
    def __init__(self, data_features, encoder_features, num_pred):
        super().__init__()
        self.data_features = Setting(data_features)
        self.encoder_features = Setting(encoder_features)
        self.num_pred = Setting(num_pred)
        self.configure()
    
    def configure(self):
        self.discriminators = nn.ModuleList([
            MLPDiscriminator(
                self.data_features, self.encoder_features, 3, start_dim=-1
                )
            for _ in range(self.num_pred)
        ])
    
    def forward(self, x, c, lengths=None):
        num_batch, max_length, cdims = c.shape
        winsize = max_length - self.num_pred - 1
        if lengths is not None:
            lengths = lengths - self.num_pred - 1
        num_neg = max(num_batch - 1, 1)
        
        xn = random_sample_sequence(x, num_neg, lengths, winsize)
        c0 = c[:, :winsize, :].contiguous()
        c0m = c0.unsqueeze(1).expand(-1, num_neg, -1, -1)
        loss = 0.0
        for k in range(1, self.num_pred + 1):
            xk = x[:, k:winsize + k, :].contiguous()
            p = self.discriminators[k-1](c0, xk)
            q = self.discriminators[k-1](c0m, xn)
            ep, en = jsd_expectations(p, q)
            if lengths is not None:
                Ep = 0.0
                En = 0.0
                for i, l in enumerate(lengths):
                    Ep += torch.sum(ep[i, :l])
                    En += torch.sum(en[i, :, :l])
                L = torch.sum(lengths)
                Ep = Ep / L
                En = En / (num_neg*L)
            else:
                Ep = torch.mean(ep)
                En = torch.mean(en)
            loss += (En - Ep)
        return loss / self.num_pred


class SeriesPredictiveCodingTrainer(BaseTrainer):
    def __init__(
            self, model, num_pred=10, model_lr=1e-3, loss_lr=1e-3,
            save_dir=None, save_interval=5, **kwargs
            ):
        super().__init__(save_dir=save_dir, save_interval=save_interval, **kwargs)
        self.model = model
        self.model.to(self.device)
        self.loss = SeriesPredictiveCodingLoss(
            model.in_features, model.out_features, num_pred
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
        return self.loss(data, encoded, lengths)
    
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
