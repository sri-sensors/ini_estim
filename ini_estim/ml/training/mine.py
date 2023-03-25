import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from .base import BaseTrainer
from ..models import BaseModel, Setting
from ..utilities import random_sample_sequence, random_sample
from ..discriminator_models import MLPDiscriminator



class MINELoss(BaseModel):
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
        
        if lengths is not None:
            Ep = 0.0
            En = 0.0
            for i, l in enumerate(lengths):
                Ep += torch.sum(p[i, :l])
                q[i, :, l:] = 0.0
            Ltotal = torch.sum(lengths)
            Ep = Ep / Ltotal
            En = torch.logsumexp(q.view(-1), 0) - math.log(Ltotal*self.samples_per_batch)
        else:
            Ep = torch.mean(p)
            En = torch.logsumexp(q.view(-1), 0) - math.log(q.numel())
        return En - Ep
    

class MINETrainer(BaseTrainer):
    def __init__(
            self, model: BaseModel, learning_rate=1e-3, save_dir=None, 
            save_interval=5, **kwargs
            ):
        super().__init__(save_dir=save_dir, save_interval=save_interval, **kwargs)
        self.model = model
        self.model.to(self.device)
        self.loss = MINELoss(model.in_features, model.out_features)
        self.loss.to(self.device)
        self.learning_rate = learning_rate
        self.optimizer = [
            optim.Adam(self.loss.parameters(), lr=self.learning_rate)
        ]
        self.negative_samples = 1
    
    def apply_optimizers(self):
        self.optimizer = [
            optim.Adam(self.loss.parameters(), lr=self.learning_rate)
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
        with torch.no_grad():
            encoded = self.model(data)  
        return self.loss(data, encoded, lengths)


class MINEDataLoss(BaseModel):
    def __init__(self, data_dim, encoder_dim, internal_factor=5):
        super().__init__()
        self.data_dim = Setting(data_dim)
        self.encoder_dim = Setting(encoder_dim)
        self.internal_factor = Setting(internal_factor)
        self.configure()
    
    def configure(self):
        self.net = MLPDiscriminator(self.data_dim, self.encoder_dim, self.internal_factor, start_dim=-1)
        
    def forward(self, data, encoded):
        batch_size, dfeat = data.shape
        _, efeat = encoded.shape
        p = self.net(encoded, data)    
        datar = random_sample(data, (batch_size,))
        q = self.net(encoded, datar)
        Ep = torch.mean(p)
        En = torch.logsumexp(q.view(-1), 0) - math.log(q.numel())
        return En - Ep
    


class MINEDataTrainer(BaseTrainer):
    def __init__(
            self, in_features, out_features, learning_rate=1e-3, save_dir=None, 
            save_interval=5, **kwargs
            ):
        super().__init__(save_dir=save_dir, save_interval=save_interval, **kwargs)
        self.model = None
        self.loss = MINEDataLoss(in_features, out_features)
        self.loss.to(self.device)
        self.learning_rate = learning_rate
        self.optimizer = [
            optim.Adam(self.loss.parameters(), lr=self.learning_rate)
        ]
        self.negative_samples = 1
    
    def apply_optimizers(self):
        self.optimizer = [
            optim.Adam(self.loss.parameters(), lr=self.learning_rate)
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
        in_data, out_data = batch
        in_data = in_data.to(self.device)
        out_data = out_data.to(self.device)
        return self.loss(in_data, out_data)

