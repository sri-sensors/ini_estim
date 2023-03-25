import torch
import torch.nn as nn
import torch.optim as optim
import abc
from tqdm.auto import tqdm, trange
from ini_estim.ml.models import BaseModel
import pathlib
import warnings
import shutil


class BaseTrainer(metaclass=abc.ABCMeta):
    """Base class for model training

    Children classes are expected to provide values for the model, loss, and 
    optimizer attributes. Before training, the train_loader must be defined. 
    To perform validation and testing, validation_loader and test_loader must 
    also be defined. If BaseTrainer.__init__ is not called, then all of the
    attributes must be defined.

    Attributes
    ----------
    model : BaseModel or torch.nn.Module
        The model to train, preferrably inheriting from BaseModel.
    loss : callable
        The loss function. This can be something simple like torch.nn.MSELoss,
        or can have its own trainable parameters, in which case it should be
        a torch.nn.Module or BaseModel (preferred).
    optimizer : torch.Optimizer or list[torch.Optimizer]
        The optimizer for model learning.
    train_loader : DataLoader
        Iterable object that returns mini-batches for training.
    test_loader : DataLoader
        Iterable object that returns mini-batches for evaluating.
    validation_loader : DataLoader
        Iterable object that returns mini-batches for validation.
    autosave : bool, optional
        Flag to auto save checkpoints every save_interval, True by default.
    save_dir : str or pathlib.Path, optional
        The directory to save checkpoints. Defaults to current directory.
    save_interval : int
        The number of epochs between saving checkpoints. This is also used as 
        the interval for performing testing and validating.
    post_epoch_callback : callable(epoch), optional.
        If defined, post_epoch_callback will be called with the epoch number
        at the end of each epoch. This can be used to stop the training
        process by returning True.

    """
    def __init__(self, save_dir="", save_interval=5, post_epoch_callback=None,
            disable_cuda=False):
        super().__init__()
        self.model = None
        self.loss = None
        self.optimizer = None
        self.train_loader = None
        self.validation_loader = None
        self.test_loader = None
        self.post_epoch_callback = post_epoch_callback
        self.autosave = True
        self.save_dir = save_dir
        self.save_interval = save_interval
        self.next_epoch = 0
        self.train_loss = []
        self.validation_loss = []
        self.test_loss = []
        self.best_train = None
        self.best_val = None
        if torch.cuda.is_available() and not disable_cuda:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
    
    @abc.abstractmethod
    def apply_optimizers(self):
        raise NotImplementedError

    def get_name(self):
        return self.__class__.__name__

    def save_checkpoint(self):
        checkpoint = {}
        checkpoint["trainer"] = self.get_name()
        checkpoint["next_epoch"] = self.next_epoch
        checkpoint["train_loss"] = self.train_loss
        checkpoint["validation_loss"] = self.validation_loss
        checkpoint["test_loss"] = self.test_loss
        save_best_train = self.best_train is None or self.train_loss[-1] < self.best_train
        if save_best_train:
            self.best_train = self.train_loss[-1]
        checkpoint["best_train"] = self.best_train
        
        just_validated = (
            len(self.validation_loss) and 
            (self.validation_loss[-1][0] == self.next_epoch - 1)
            )
        save_best_val = just_validated and (
            self.best_val is None or 
            self.validation_loss[-1][1] < self.best_val[1]
        )
        if save_best_val:
            self.best_val = self.validation_loss[-1]
        checkpoint["best_val"] = self.best_val
        
        # Save the model dict
        checkpoint["model"] = self.get_model_checkpoint()

        # Save the loss if it has a state
        checkpoint["loss"] = self.get_loss_checkpoint()
        
        # Save the optimizer state
        checkpoint["optimizers"] = self.get_optimizer_checkpoint()

        # Save any custom information
        checkpoint["meta"] = self.get_meta_checkpoint()
        
        save_dir = pathlib.Path(self.save_dir)
        savepath = save_dir / "{}_last.pt".format(self.get_name())
        if pathlib.Path(savepath).exists():
            pname = savepath.name.replace("_last", "_previous")
            previous = save_dir / pname
            shutil.copy2(savepath, previous)
        torch.save(checkpoint, savepath)

        if save_best_train:
            savepath = save_dir / "{}_best_train.pt".format(self.get_name())
            torch.save(checkpoint, savepath)
        
        if save_best_val:
            savepath = save_dir / "{}_best_val.pt".format(self.get_name())
            torch.save(checkpoint, savepath)
        
        return savepath

    def load_checkpoint(self, checkpoint):
        if not isinstance(checkpoint, dict):
            checkpoint = torch.load(checkpoint)
        
        if checkpoint["trainer"] != self.get_name():
            warnings.warn(
                "Trainer name mismatch \"{}\" != \"{}\"".format(
                    checkpoint["trainer"], self.get_name()
                )
            )
        self.next_epoch = checkpoint["next_epoch"]
        self.train_loss = checkpoint["train_loss"]
        self.validation_loss = checkpoint["validation_loss"]
        self.test_loss = checkpoint["test_loss"]
        self.best_train = checkpoint["best_train"]
        self.best_val = checkpoint["best_val"]
        self.restore_model(checkpoint["model"])
        self.restore_loss(checkpoint.get("loss"))
        self.restore_optimizers(checkpoint["optimizers"])
        self.restore_meta_checkpoint(checkpoint.get("meta", None))
    
    def restore_model(self, model_checkpoint: dict):
        if model_checkpoint is None:
            return
        if "config" in model_checkpoint:
            self.model.load_checkpoint(model_checkpoint)
        else:
            if model_checkpoint["name"] != self.model._get_name():
                warnings.warn("Model name mismatch \"{}\" != \"{}\"".format(
                    model_checkpoint["name"], self.model._get_name()
                ))
            self.model.load_state_dict(model_checkpoint["state_dict"])
        self.model.to(self.device)

    def get_model_checkpoint(self):
        if isinstance(self.model, BaseModel):
            return self.model.to_dict()
        elif isinstance(self.model, nn.Module):
            return {
                "name": self.model._get_name(),
                "state_dict": self.model.state_dict()   
            }
        else:
            return None

    def restore_loss(self, loss_checkpoint: dict):
        if loss_checkpoint is None:
            return
        if isinstance(self.loss, BaseModel):
            self.loss.load_checkpoint(loss_checkpoint)
        elif isinstance(self.loss, nn.Module):
            if loss_checkpoint["name"] != self.loss._get_name():
                warnings.warn("Loss name mismatch \"{}\" != \"{}\"".format(
                    loss_checkpoint["name"], self.loss._get_name()
                ))
            self.loss.load_state_dict(loss_checkpoint["state_dict"])
        self.loss.to(self.device)
    
    def get_loss_checkpoint(self):
        if isinstance(self.loss, BaseModel):
            return self.loss.to_dict()
        elif isinstance(self.loss, nn.Module):
            return {
                "name": self.loss._get_name(),
                "state_dict": self.loss.state_dict()   
            }
        else:
            return None
        
    def restore_optimizers(self, optimizer_checkpoint):
        self.apply_optimizers()
        if isinstance(self.optimizer, list):
            for o, state in zip(self.optimizer, optimizer_checkpoint):
                o.load_state_dict(state)        
        else:
            self.optimizer.load_state_dict(optimizer_checkpoint)

    def get_optimizer_checkpoint(self):
        if isinstance(self.optimizer, list):
            return [o.state_dict() for o in self.optimizer]
        else:
            return self.optimizer.state_dict()        

    def restore_meta_checkpoint(self, meta_checkpoint):
        return None

    def get_meta_checkpoint(self):
        return None

    def train(self, epochs, show_progress=True):
        epoch_it = trange(epochs) if show_progress else range(epochs)
        stop_training = False
        num_train = len(self.train_loader)

        for epoch in epoch_it:
            running_loss = 0.0
            if self.model is not None:
                self.model.train()
            for batch in self.train_loader:
                current_loss = self.train_step(batch)
                running_loss += current_loss
            
            self.train_loss.append(running_loss / num_train)
            self.next_epoch += 1
            if self.save_interval and not (epoch + 1) % self.save_interval:
                if self.validation_loader is not None:
                    if show_progress:
                        epoch_it.set_description("Validating...")
                    vloss = self.validate()
                    self.validation_loss.append((self.next_epoch - 1, vloss))
                
                if self.test_loader is not None:
                    if show_progress:
                        epoch_it.set_description("Testing...")
                    tloss = self.test()
                    self.test_loss.append((self.next_epoch - 1, tloss))

                if self.autosave:
                    if show_progress:
                        epoch_it.set_description("Saving...")
                    self.save_checkpoint()

            if show_progress:
                if len(self.train_loss) > 1:
                    diff = self.train_loss[-1] - self.train_loss[-2]
                    desc = "{:>6g}({:>4g})".format(self.train_loss[-1], diff)
                else:
                    desc = "{:>12g}".format(self.train_loss[-1])
                epoch_it.set_description(desc)
            

            if self.post_epoch_callback is not None and \
                    self.post_epoch_callback(self.next_epoch - 1):
                break
        
        if show_progress:
            epoch_it.close()
        
        if self.autosave:
            self.save_checkpoint()
            
    def test(self):
        """ Runs evaluate with test_loader """
        return self.evaluate(self.test_loader)
    
    def validate(self):
        """ Runs evaluate with validation_loader """
        return self.evaluate(self.validation_loader)
    
    def evaluate(self, data_loader):
        """ Gets average loss for data in data_loader

        Sets model to eval() mode and runs with torch.no_grad().

        Parameters
        ----------
        data_loader : DataLoader
            Batched dataset to run model on.

        Returns
        -------
        float
            Average loss value for all batches in data_loader
        """
        if self.model is not None:
            self.model.eval()
        with torch.no_grad():
            loss = 0.0
            for batch in data_loader:
                loss += self.eval_step(batch)
        return loss / len(data_loader)

    @abc.abstractmethod
    def train_step(self, batch):
        """ Single step of training.

        Classes inheriting from BaseTrainer must implement this method.
        Typical steps are:
        1. Compute model output
        2. Compute loss
        3. loss.backward()
        4. optimizers.step()
        5. zero gradients

        Parameters
        ----------
        batch
            Batched data from train_loader to perform training step.

        Returns
        -------
        float
            Loss value for training step, usually from loss.item().
        """
        raise NotImplementedError

    @abc.abstractmethod
    def eval_step(self, batch):
        """ Single step of evaluation.

        Classes inheriting from BaseTrainer must implement this method.
        eval_step is called from evaluate within torch.no_grad() context and 
        model set to eval() mode.

        Parameters
        ----------
        batch
            Batched data from a DataLoader to perform evaluation step.

        Returns
        -------
        float
            Loss value for evaluation step, usually from loss.item().
        """
        raise NotImplementedError

