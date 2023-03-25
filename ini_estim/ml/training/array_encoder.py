import torch
from ..models import BaseModel, model_from_checkpoint
from .base import BaseTrainer
from .direct_infomax import DirectInfoMaxLoss
import torch.optim as optim
from ..electrode_array import ElectrodeArray



class ElectrodeArrayDIMTrainer(BaseTrainer):
    def __init__(
            self, ea_model: BaseModel, electrode_array_type, 
            num_sampling_points=500, signal_map=None, input_scale=1.0,
            pre_encoder_model: BaseModel=None, force_positive_current=False, model_lr=1e-3, 
            loss_lr=1e-3, save_dir=None, save_interval=5, **kwargs
            ):
        super().__init__(save_dir=save_dir, save_interval=save_interval, **kwargs)
        self.signal_map = signal_map
        self.pre_encoder = pre_encoder_model
        if self.pre_encoder is not None:
            self.pre_encoder.to(self.device)
            self.pre_encoder.train(False)
        self.ea_type = electrode_array_type
        self.ea = ElectrodeArray(electrode_array_type, num_sampling_points, 
            input_scale=input_scale, positive_current_only=force_positive_current)
        self.ea.to(self.device)
        self.model = ea_model
        self.model.to(self.device)
        self.loss = DirectInfoMaxLoss(
            ea_model.in_features, self.ea.out_features
        )
        self.loss.to(self.device)
        self.model_lr = model_lr
        self.loss_lr = loss_lr
        self.apply_optimizers()
        self.negative_samples = 1

    def apply_optimizers(self):
        self.optimizer = [
            optim.Adam(self.model.parameters(), lr=self.model_lr),
            optim.Adam(self.loss.parameters(), lr=self.loss_lr)
        ]

    def restore_meta_checkpoint(self, meta_checkpoint):
        self.signal_map = meta_checkpoint['signal_map']
        self.negative_samples = meta_checkpoint['negative_samples']
        self.pre_encoder = model_from_checkpoint(meta_checkpoint['pre_encoder'])
        if self.pre_encoder is not None:
            self.pre_encoder.to(self.device)
            self.pre_encoder.train(False)

        self.ea = ElectrodeArray(
            meta_checkpoint['ea_type'],
            meta_checkpoint['num_sampling_points']
        )
        self.ea.to(self.device)

    def get_meta_checkpoint(self):
        pre_encoder_dict = None if self.pre_encoder is None else self.pre_encoder.to_dict()
        out = dict(
            ea_type=self.ea_type,
            num_sampling_points=self.ea.out_features,
            pre_encoder=pre_encoder_dict,
            signal_map=self.signal_map,
            negative_samples=self.negative_samples
        )
        return out

    def get_loss(self, batch):
        data = batch[0]
        if len(batch) > 2:
            lengths = batch[1]
        else:
            lengths = None

        data = data.to(self.device)
        if self.pre_encoder is not None:
            with torch.no_grad():
                data = self.pre_encoder(data)
        
        out = self.ea(self.model(data))
        self.loss.samples_per_batch = self.negative_samples
        return self.loss(data, out, lengths)

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
