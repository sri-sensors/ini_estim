import torch
import torch.nn as nn
import torch.optim as optim
from .base import BaseTrainer
from ..models import model_from_checkpoint, BaseModel


class PredictorTrainer(BaseTrainer):
    def __init__(
            self, decoder_model: BaseModel, encoder_model: BaseModel, 
            lookahead=1, lr=1e-3, save_dir='', save_interval=5, **kwargs
            ):
        super().__init__(save_dir=save_dir, save_interval=save_interval, **kwargs)
        self.model = decoder_model
        self.model.to(self.device)
        self.encoder = encoder_model
        if self.encoder is not None:
            self.encoder.to(self.device)
        self.loss = nn.MSELoss(reduction='sum')
        self.loss.to(self.device)
        self.lookahead = lookahead
        self.lr = lr
        self.apply_optimizers()

    def apply_optimizers(self):
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
    
    def restore_meta_checkpoint(self, meta_checkpoint):
        self.encoder = model_from_checkpoint(meta_checkpoint["encoder"])
        if self.encoder is not None:
            self.encoder.to(self.device)
        self.lookahead = meta_checkpoint["lookahead"]

    def get_meta_checkpoint(self):
        encoder_dict = None if self.encoder is None else self.encoder.to_dict()
        out = dict(encoder=encoder_dict, lookahead=self.lookahead)
        return out

    def get_loss(self, batch):
        if len(batch) == 3:
            data, lengths, _ = batch
        else:
            data, _ = batch
            lengths = None
        data = data.to(self.device)
        if self.encoder is not None:
            with torch.no_grad():
                encoded = self.encoder(data)
        else:
            encoded = data
        decoded = self.model(encoded)
        off = self.lookahead
        if lengths is not None:
            loss = 0.0
            for i, l in enumerate(lengths):
                loss += self.loss(decoded[i, :l-off], data[i, off:l])
            N = torch.sum(lengths)*data.shape[2]
        else:
            l = data.shape[1]
            loss = self.loss(decoded[:, :l-off], data[:, off:l])
            N = data.shape[0]*data.shape[1]*data.shape[2]
        return loss/N
    
    def train_step(self, batch):
        loss = self.get_loss(batch)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss.item()
    
    def eval_step(self, batch):
        return self.get_loss(batch).item()
