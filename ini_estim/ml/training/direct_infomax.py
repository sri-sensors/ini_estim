import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from .base import BaseTrainer
from ..models import BaseModel, Setting
from ..utilities import random_sample_sequence
from ..discriminator_models import MLPDiscriminator


def jsd_expectations(p, q):
    log_2 = math.log(2.0)
    ep = log_2 - F.softplus(-p)
    en = F.softplus(-q) + q - log_2
    return ep, en


class DirectInfoMaxLoss(BaseModel):
    def __init__(self, data_dim, encoder_dim, samples_per_batch=1, internal_factor=3):
        super().__init__()
        self.data_dim = Setting(data_dim)
        self.encoder_dim = Setting(encoder_dim)
        self.internal_factor = Setting(internal_factor)
        self.samples_per_batch = Setting(samples_per_batch)
        self.configure()
    
    def configure(self):
        self.net = MLPDiscriminator(self.data_dim, self.encoder_dim, self.internal_factor, start_dim=-1)
        
    def forward(self, data, encoded, lengths=None):
        batch_size, max_length, dfeat = data.shape
        _, _, efeat = encoded.shape
        p = self.net(encoded, data)
        datar = random_sample_sequence(data, self.samples_per_batch, lengths)
        encoded_ex = encoded.unsqueeze(1).expand(
            -1, self.samples_per_batch, -1, -1
            )
        q = self.net(encoded_ex, datar)
        ep, en = jsd_expectations(p, q)
        if lengths is not None:
            Ep = 0.0
            En = 0.0
            for i, l in enumerate(lengths):
                Ep += (torch.sum(ep[i, :l]))
                En += (torch.sum(en[i, :, :l]))
            Ltotal = torch.sum(lengths)
            Ep = Ep / Ltotal
            En = En / (Ltotal*self.samples_per_batch)
        else:
            Ep = torch.mean(ep)
            En = torch.mean(en)
        return En - Ep
    

class DirectInfoMaxTrainer(BaseTrainer):
    def __init__(
            self, model: BaseModel, model_lr=1e-3, loss_lr=1e-3, save_dir=None, 
            save_interval=5, **kwargs
            ):
        super().__init__(save_dir=save_dir, save_interval=save_interval, **kwargs)
        self.model = model
        self.model.to(self.device)
        self.loss = DirectInfoMaxLoss(model.in_features, model.out_features)
        self.loss.to(self.device)
        self.model_lr = model_lr
        self.loss_lr = loss_lr
        self.optimizer = [
            optim.Adam(self.model.parameters(), lr=model_lr),
            optim.Adam(self.loss.parameters(), lr=loss_lr)
        ]
        self.negative_samples = 1
    
    def apply_optimizers(self):
        self.optimizer = [
            optim.Adam(self.model.parameters(), lr=self.model_lr),
            optim.Adam(self.loss.parameters(), lr=self.loss_lr)
        ]

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

    def get_loss(self, batch):
        if len(batch) == 3:
            data, lengths, _ = batch
        else:
            data, _ = batch
            lengths = None
        data = data.to(self.device)
        self.loss.samples_per_batch = self.negative_samples
        encoded = self.model(data)  
        return self.loss(data, encoded, lengths)

