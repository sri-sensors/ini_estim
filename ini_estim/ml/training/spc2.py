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


class Bilinear2(nn.Module):
    def __init__(self, in1, in2, nint, out):
        super().__init__()
        self.bilayer = nn.Bilinear(in1, in2, nint)
        self.llayer = nn.Linear(nint, out)
    
    def forward(self, x1, x2):
        return self.llayer(torch.relu(self.bilayer(x1, x2)))



class Bilinear3(nn.Module):
    def __init__(self, in1, in2, out=1):
        super().__init__()
        self.a = nn.Linear(in1, out, bias=False)
        self.b = nn.Linear(in2, out, bias=False)
        self.c = nn.Bilinear(in1, in2, out, bias=True)
    
    def forward(self, x1, x2):
        return self.a(x1) + self.b(x2) + self.c(x1, x2)


class SeriesPredictiveCodingLoss2(BaseModel):
    def __init__(self, data_features, encoder_features, num_pred, 
            discriminator_type="mlp", include0=False):
        super().__init__()
        self.data_features = Setting(data_features)
        self.encoder_features = Setting(encoder_features)
        self.num_pred = Setting(num_pred)
        self.discriminator_type = Setting(discriminator_type)
        self.include0 = Setting(include0)
        self.configure()

    def configure(self):
        d = self.discriminator_type.lower()
        if self.include0:
            f = self.data_features*(self.num_pred + 1)
        else:
            f = self.data_features*self.num_pred

        if d == "mlp":
            self.discriminator = MLPDiscriminator(
                f, self.encoder_features, 3, start_dim=-1
            )
        elif d == "bilinear":
            self.discriminator = nn.Bilinear(self.encoder_features, f, 1)
        elif d == "bilinear2":
            self.discriminator = Bilinear2(self.encoder_features, f, 2*f, 1)
        else:
            self.discriminator = Bilinear3(self.encoder_features, f, 1)
                
    def forward(self, x: torch.Tensor, c: torch.Tensor, lengths=None):
        num_batch, max_length, cdims = c.shape
        num_neg = max(num_batch - 1, 1)
        offset = 1 if self.include0 else 0
        if lengths is not None:
            lengths = lengths - self.num_pred + offset
        winsize = max_length - self.num_pred
        xn = x.unfold(-2, self.num_pred + offset, 1).flatten(start_dim=-2)
        xn = xn[:,1-offset:].contiguous()

        cw = c[:, :winsize].contiguous()
        xr = random_sample_sequence(xn, num_neg)
        cwr = cw.unsqueeze(1).expand(-1, num_neg, -1, -1).contiguous()
        p = self.discriminator(cw, xn)
        q = self.discriminator(cwr, xr)
        p = math.log(2.0) - F.softplus(-p)
        q = F.softplus(-q) + q - math.log(2.0)
        if lengths is not None:
            ep = 0.0
            eq = 0.0
            for i, l in enumerate(lengths):
                ep += torch.sum(p[i, :l])
                eq += torch.sum(q[i, :l])
            L = max(torch.sum(lengths), 1.0)
            ep = ep/L
            eq = ep/(L*num_neg)
        else:
            ep = torch.mean(p)
            eq = torch.mean(q)
        return eq - ep


class SeriesPredictiveCodingTrainer2(BaseTrainer):
    def __init__(
            self, model, discriminator="mlp", num_pred=10, include0=False,
            model_lr=1e-3, loss_lr=1e-3, save_dir=None, save_interval=5,
            **kwargs
            ):
        super().__init__(save_dir=save_dir, save_interval=save_interval, **kwargs)
        self.model = model
        self.model.to(self.device)
        self.loss = SeriesPredictiveCodingLoss2(
            model.in_features, model.out_features, num_pred, discriminator,
            include0=include0
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
