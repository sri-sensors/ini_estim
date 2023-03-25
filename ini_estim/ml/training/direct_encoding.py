import torch
import torch.optim as optim
import numpy as np
from tqdm.auto import tqdm, trange
from ini_estim.datasets import get_dataset
import pathlib
import json


def restore_encoder(config_file, checkpoint_file, in_features=None, data_folder=None):
    """ Restore a previously saved encoder

    Parameters
    ----------
    config_file : str or pathlib.Path
        Location of the JSON file describing the encoder
    checkpoint_file : str or pathlib.Path
        Location of the checkpoint file containing the encoder state
    in_features : int
        The number of input features for the encoder. If unspecified,
        restore_encoder will try to read this value from the config file, 
        and as a last resort will load the dataset. 
    data_folder : str or pathlib.Path
        Location of dataset specified in config file. Set to None to use
        the same location as is in the config file. This is only necessary 
        if the number of input features is not specified in the config file.
    Returns
    -------
    encoder : nn.Module
    config : dict

    """
    with open(config_file, 'r') as f:
        cfg = json.load(f)
    
    if in_features is None:
        in_features = cfg.get("in_features")
    out_features = cfg.get("dimensions")
    encoder_name = cfg.get("model")

    if in_features is None:
        # Must load dataset to determine in features
        data_folder = data_folder if data_folder is not None else cfg["input_folder"]
        _, _, _, in_features, _, _ = get_dataset(cfg["dataset"],  data_folder)

    encoder = get_encoder(encoder_name, in_features, out_features, cfg)
    checkpoint = torch.load(checkpoint_file)
    encoder.load_state_dict(checkpoint["encoder_state_dict"])
    return encoder, cfg


def get_encoder(encoder_name, in_features, out_features, args):
    """ Convenience function for generating an encoder model.

    Parameters
    ----------
    encoder_name : str
        The name of the encoder type. Accepted options are "esn" or "mlp"
    in_features : int
        The number of features for the encoder input (size of last dimension)
    out_features : int
        The number of features for the encoder output
    args : dict
        Dictionary of arguments for the encoder. Each encoder type accepts 
        different arguments.
        esn 
            "hidden_units" - the number of nodes in the reservoir 
            "spectral_radius" - the spectral radius of the reservoir
            "density" - the percentage of non-zero reservoir weights
            "leak_rate" - the leaky integration rate (1.0 = no integration of 
                previous states.)
        mlp
            "hidden_units" - the number of nodes in the hidden layer.


    Returns
    -------
    torch.nn.Module
        The PyTorch module that implements the specified encoder.

    Raises
    ------
    NotImplementedError
        This error is raised if an invalid encoder is specified.
    """
    encoder_name = encoder_name.lower()
    model = None
    if encoder_name == "esn":
        from ini_estim.ml.models import esn
        model = esn.ESN(
            in_features, out_features, args["hidden_units"],
            spectral_radius=args["spectral_radius"], density=args["density"],
            leak_rate=args["leak_rate"]
            )
    elif encoder_name == "mlp":
        from ini_estim.ml.networks import MLP
        if args.get("hidden_units") is None:
            hidden_features = (in_features + out_features) // 2
        else:
            hidden_features = args["hidden_units"]
        model = MLP([in_features, hidden_features, out_features], flatten=False)
    else:
        raise NotImplementedError(
            "Specified encoder type \"{}\" has not been implemented.".format(
                encoder_name
            ))
    return model


def train_mi_encoder(
        encoder_model, loss_model, train_loader, validation_loader=None,  
        test_loader=None, num_epochs=50, use_labels=False, stop_early=False, 
        save_dir=None, validation_interval=5, encoder_optimizer=optim.Adam, 
        enc_opt_params=None, loss_optimizer=optim.Adam, loss_opt_params=None,
        callback=None, subpbar=True
    ):
    """ Train an encoder to maximize MI using the MINE/DeepInfoMax method

    Parameters
    ----------
    encoder_model : nn.Module
        The encoder to train.
    loss_model : nn.Module
        The loss_model is the discriminator network for optimizing the 
        variational loss function (e.g. Jensen-Shannon Divergence). It 
        takes as its input (encoder_output, in_class_data, out_of_class_data)
        to calculate the loss value. The parameters of loss_model will 
        be optimized at each iteration using loss_optimizer.
    train_loader : DataLoader
        Data loader for training data batches. Each iteration of train_loader
        should return a tuple of (data, labels), where data is a Tensor with 
        the size of (batch_size, sequence_length, num_features), and labels
        is a Tensor with the size (batch_size, ) assigning a label to each
        sequence.
    validation_loader : DataLoader (Optional)
        Data loader for validation data batches with the same form as 
        train_loader. Validation is tested every validation_interval epochs. 
        Set to None (default) if not using a validation set.
    test_loader : DataLoader (Optional)
        Data loader for test data. Similar to validation_loader or train_loader.
        Set to None (default) if not using a test set. Otherwise, at the end
        of training, the loss will be computed on the test set and returned.
    num_epochs : int (Optional)
        The maximum number of epochs to complete (default: 50).
    use_labels : bool (Optional)
        Boolean for whether to use labels when specifying "out of class"
        data in the MI estimation. If True, then the "out of class" data
        must come from different labels than the encoded target. If False
        (default), then the "out of class" data is randomly sampled from
        the batch. WARNING: Not yet implemented.
    stop_early : bool (Optional)
        Boolean to specify whether to stop early when the validation data
        loss starts to increase. Default is False. If no validation_loader 
        is specified, this option is ignored.
    save_dir : str (Optional)
        Directory to save intermediate and final results. Set to None (default)
        if not required. Because the saved files have generic names, this 
        directory should be unique.
    validation_interval : int (Optional)
        Interval (# of epochs) to run validation (if validation_loader is not 
        None) or save intermediate results (if save_dir is not None).
        Default interval is 5.
    encoder_optimizer : Optimizer (Optional)
        The optimizer for the encoder to use. Default is Adam.
    encoder_params : dict (Optional)
        Additional keyword arguments to pass to the optimizer.
    loss_optimizer: Optimizer (Optional)
        The optimizer for the loss_model to use. Default is Adam
    loss_params : dict (Optional)
        Additional keyword arguments to pass to the loss optimizer.
    callback : Callable[int, nn.Module] (Optional)
        Callback function called at the end of each epoch - takes the epoch
        number and encoder model as arguments.
    subpbar : bool (Optional)
        Boolean to display progress bar for sub iterations, by default True.
    Returns
    -------
    loss_history
        The loss at each epoch of training
    validation_loss_history
        The loss on the validation set at each validation_interval
    test_loss
        The final loss on the test set.
    """
    enc_opt_params = {} if enc_opt_params is None else enc_opt_params
    enc_opt = encoder_optimizer(encoder_model.parameters(), **enc_opt_params)
    loss_opt_params = {} if loss_opt_params is None else loss_opt_params
    loss_opt = loss_optimizer(loss_model.parameters(), **loss_opt_params)
    
    encoder_model.train(True)
    loss_model.train(True)

    loss_history = []
    validation_loss_history = []
    test_loss_history = []
    batch_size = train_loader.batch_size
    num_batches = len(train_loader)
    
    rng = np.random.default_rng()
    def get_loss(data, encoder_model, loss_model):
        # generate 'out of class' data set
        # Note: In the DeepInfoMax paper, they use all out-of-class examples
        # To save on computation, we just use a sub-sample by rolling the
        # indexes a random amount (making sure not to roll over to the 
        # original index).
        nonlocal rng
        idx_seq = np.arange(len(data))
        idx = np.roll(idx_seq, rng.integers(1, len(data)))
        dprime = data[idx]

        # encode and get loss
        y = encoder_model(data)
        return loss_model(y, data, dprime)
    
    def get_data(batch):
        """ Helper function to extract data that may not have labels """
        if not isinstance(batch, torch.Tensor):
            data, labels = batch
        else:
            data = batch
            labels = None
            use_labels = False
        return data

    epoch_iterator = trange(num_epochs)
    for epoch in epoch_iterator:
        
        encoder_model.train()
        loss_model.train()
        running_loss = 0.0
        if subpbar:
            batch_iterator = tqdm(train_loader, total=num_batches, leave=False)
        else:
            batch_iterator = train_loader
        for i, batch in enumerate(batch_iterator):
            
            enc_opt.zero_grad()
            loss_opt.zero_grad()

            data = get_data(batch)
            loss = get_loss(data, encoder_model, loss_model)

            loss.backward()
            loss_opt.step()
            enc_opt.step()
            running_loss += loss.item()
            if subpbar:
                batch_iterator.set_description(
                    "Running loss: {:<12g}".format(running_loss/(i+1))
                    )
        
        loss_history.append(running_loss / num_batches)
        epoch_iterator.set_description(
            "Current loss: {:<12g}".format(loss_history[-1])
        )
        
        if callback:
            callback(epoch, encoder_model)

        if (epoch+1) % validation_interval and epoch < num_epochs - 1:
            continue
        
        encoder_model.eval()
        loss_model.eval()
        if validation_loader:
            vbs = validation_loader.batch_size
            nvb = len(validation_loader)
            vbatch_it = tqdm(validation_loader, total=nvb, leave=False)
            vbatch_it.set_description("Validating")
            vloss = 0.0
            n = 0
            for batch in vbatch_it:
                data = get_data(batch)
                with torch.no_grad():
                    loss = get_loss(data, encoder_model, loss_model)
                vloss += loss.item()*data.shape[0]
                n += data.shape[0]
            vloss /= n
            validation_loss_history.append(vloss)
        
        if test_loader:
            test_it = tqdm(test_loader, total=len(test_loader), leave=False)
            test_it.set_description("Evaluating test data")
            test_loss = 0.0
            encoder_model.eval()
            loss_model.eval()
            n = 0
            for batch in test_it:
                data = get_data(batch)
                with torch.no_grad():
                    l = get_loss(data, encoder_model, loss_model).item()
                    test_loss += l*data.shape[0]
                n += data.shape[0]
            test_loss /= n
            test_loss_history.append(test_loss)

        if save_dir:
            save_checkpoint(
                save_dir, epoch, encoder_model, loss_model, enc_opt,
                loss_opt, loss_history, validation_loss_history, test_loss_history
                )
        
        if stop_early and len(validation_loss_history) >= 2:
            if validation_loss_history[-2] > validation_loss_history[-1]:
                print("Validation loss has increased. Stopping!")
                break


    return loss_history, validation_loss_history, test_loss_history


def save_checkpoint(
        save_dir, epoch, encoder_model, loss_model, encoder_optimizer, 
        loss_optimizer, loss_history, validation_loss_history, test_loss_history):
    """ Save checkpoint for model recovery 
    
    This actually saves 2 versions of the checkpoint. One version is for
    just recovering the encoder, which has the epoch number appended to the
    filename. The other version is an optimization checkpoint, which includes
    everything needed to resume optimization and will overwrite the previous
    optimization checkpoint. 
    """
    checkpoint = {
        "epoch": epoch + 1,
        "encoder": str(encoder_model),
        "encoder_state_dict": encoder_model.state_dict(),
        "loss_model": str(loss_model),
        "loss_state_dict": loss_model.state_dict(),
        "loss_history": loss_history,
        "validation_loss_history": validation_loss_history,
        "test_loss_history": test_loss_history
    }
    save_dir = pathlib.Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    filename = "encoder_epoch{}.pt".format(epoch+1)
    torch.save(checkpoint, save_dir / filename)
    
    # overwrite the last optimization checkpoint
    checkpoint["encoder_optimizer"] = str(encoder_optimizer)
    checkpoint["encoder_optimizer_state_dict"] = encoder_optimizer.state_dict()
    checkpoint["loss_optimizer"] = str(loss_optimizer)
    checkpoint["loss_optimizer_state_dict"] = loss_optimizer.state_dict()
    filename = "full_optimization_checkpoint.pt"
    torch.save(checkpoint, save_dir / filename)
