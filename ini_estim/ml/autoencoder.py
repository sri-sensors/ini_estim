import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm, trange
import pathlib
from ini_estim.ml.training.direct_encoding import get_encoder
from ini_estim.ml.decoder import MLPDecoder 


def get_autoencoder(cfg, state_dict=None):
    encoder = get_encoder(
        cfg["model"], cfg["in_features"], cfg["dimensions"], cfg
        )
    
    Nin = cfg["in_features"]*cfg["num_samples"]
    N_decoder = cfg["dimensions"]*cfg["num_samples"]
    decoder = MLPDecoder(
        N_decoder, [cfg["num_samples"], cfg["in_features"]], cfg["hidden_unit_factor"]
    )
    model = nn.Sequential(encoder, decoder)
    if state_dict is not None:
        model.load_state_dict(state_dict)
    return model


def train_autoencoder(
        model, train_loader, validation_loader=None,
        test_loader=None, num_epochs=50, save_dir=None, validation_interval=5,
        stop_early=False, optimizer=optim.Adam, opt_params=None
    ):
    """ Train a decoder for a pre-trained encoder.

    Parameters
    ----------
    model : nn.Module
        The autoencoder to train
    train_loader : DataLoader
        Data loader for training data batches. 
    validation_loader : DataLoader, optional
        Data loader for validation, by default None. If specified, validation is 
        performed every validation_interval epochs.
    test_loader : DataLoader, optional
        Data loader for test, by default None. Similar to validation_loader.
    num_epochs : int, optional
        The number of training epochs to complete, by default 50.
    save_dir : str, optional
        If specified, the directory to save intermediate and final results, by 
        default None.
    validation_interval : int, optional
        Interval (# of epochs) to run validation, test, and save intermediate 
        results, by default 5.
    stop_early : bool, optional
        Boolean to specify whether to stop early when the validation data
        loss starts to increase. Default is False. If no validation_loader 
        is specified, this option is ignored.
    optimizer : Optimizer, optional
        The optimizer to use, Adam by default.
    opt_params : dict, optional
        Additional keyword arguments to pass to the optimizer, by default None

    Returns
    -------
    loss_history, validation_loss_history, test_loss
    """
    opt_params = {} if opt_params is None else opt_params
    model.train()
    
    loss_history = []
    validation_loss_history = []
    test_loss_history = []
    batch_size = train_loader.batch_size
    num_batches = len(train_loader)

    opt = optimizer(model.parameters(), **opt_params)
    loss_fun = nn.MSELoss()

    def get_data(batch):
        if not isinstance(batch, torch.Tensor):
            data, _ = batch
        else:
            data = batch
        return data

    epoch_it = trange(num_epochs)
    for epoch in epoch_it:
        model.train()
        running_loss = 0.0
        batch_it = tqdm(train_loader, total=len(train_loader), leave=False)
        for i, batch in enumerate(batch_it):
            opt.zero_grad()
            data = get_data(batch)
            decoded = model(data)
            loss = loss_fun(data, decoded)
            loss.backward()
            opt.step()
            running_loss += loss.item()
            batch_it.set_description(
                "Running loss: {:<12g}".format(running_loss/(i+1))
            )
        loss_history.append(running_loss / len(train_loader))

        if (epoch+1) % validation_interval and epoch < num_epochs - 1:
            continue

        if validation_loader:
            val_iter = tqdm(validation_loader, total=len(validation_loader), leave=False)
            val_iter.set_description("Validating")
            vloss = 0.0
            model.eval()
            for batch in val_iter:
                data = get_data(batch)
                with torch.no_grad():
                    decoded = model(data)
                    loss = loss_fun(data, decoded)
                    vloss += loss.item()
            validation_loss_history.append(vloss / len(validation_loader))
        
        if test_loader:
            test_iter = tqdm(test_loader, total=len(test_loader), leave=False)
            test_iter.set_description("Testing")
            test_loss = 0.0
            n = 0
            model.eval()
            for batch in test_iter:
                data = get_data(batch)
                with torch.no_grad():
                    decoded = model(data)
                    loss = loss_fun(data, decoded)
                    test_loss += loss.item()
                
            test_loss_history.append(test_loss / len(test_loader))
        
        if save_dir:
            checkpoint = {
                "epoch": epoch + 1,
                "model": str(model),
                "state_dict": model.state_dict(),
                "loss_history": loss_history,
                "validation_loss_history": validation_loss_history,
                "test_loss_history": test_loss_history,
            }
            save_dir = pathlib.Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            filename = "autoencoder_epoch{}.pt".format(epoch + 1)
            torch.save(checkpoint, save_dir / filename)
        
        if stop_early and len(validation_loss_history) >= 2:
            if validation_loss_history[-2] > validation_loss_history[-1]:
                print("Validation loss has increased. Stopping!")
                break
    
    return loss_history, validation_loss_history, test_loss_history
