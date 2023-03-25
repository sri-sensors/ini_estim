from . import utah
from .base import BaseModel, Setting, OUTPUT_ACTIVATIONS
from .mlp import MLP
from .cnn import TimeSeriesCNN, TimeSeriesMultiCNN
from .esn import ESN, ESNCell
from .utah import BiomimeticEncoder
from .linear import Linear, Diagonal
import torch


def get_model(model_name):
    models = BaseModel.subclasses()
    return models[model_name]


def model_from_checkpoint(checkpoint):
    import torch

    if checkpoint is None:
        return None 
    if not isinstance(checkpoint, dict):
        checkpoint = torch.load(checkpoint)
    try:
        ModelClass = get_model(checkpoint["name"])
    except KeyError:
        print(checkpoint.keys())
        raise
    return ModelClass.from_checkpoint(checkpoint)
