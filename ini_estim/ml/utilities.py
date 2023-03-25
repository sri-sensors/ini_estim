import torch
from torch.nn.utils.rnn import pad_sequence
from tqdm.auto import trange, tqdm


def trainer(
        data_loader, train_step_callback, epochs, epoch_callback=None, 
        show_pbar=True
    ):
    """Convenience function for running a training loop

    Parameters
    ----------
    data_loader : DataLoader
        Training data loader
    train_step_callback : callable
        Function for doing the actual training. Takes item from data_loader
        as an argument and returns the current loss as a float.
    epochs : int
        The number of epochs to train
    epoch_callback : callable(epoch : int)
        Function to call at the end of each epoch, by default None. 
        epoch_callback takes the epoch as an argument and returns None or False  
        if training shall continue, and True if training should stop. 
    show_pbar : bool, optional
        Flag to show tqdm progress bar, by default True

    Returns
    -------
    list of floats
        Loss value at each epoch
    """ 
    epoch_it = trange(epochs) if show_pbar else range(epochs)
    loss_history = []
    stop_training = False
    for epoch in epoch_it:
        running_loss = 0.0
        for batch in data_loader:
            current_loss = train_step_callback(batch)
            running_loss += current_loss
        
        loss_history.append(running_loss / len(data_loader))

        if show_pbar:
            if len(loss_history) > 1:
                diff = loss_history[-1] - loss_history[-2]
                desc = "{:>6g}({:>4g})".format(loss_history[-1], diff)
            else:
                desc = "{:>12g}".format(loss_history[-1])
            epoch_it.set_description(desc)

        if epoch_callback is not None and epoch_callback(epoch):
            break
    
    return loss_history


def padder(batch):
    """ Pad data in batch to same length

    Parameters
    ----------
    batch : iterable
        Iterable of sequences for padding. Each entry is assumed to consist of 
        two elements (sequence, label). Sequences will be sorted by length
        before padding.

    Returns
    -------
    torch.Tensor
        Padded sequences of size (batch_size, max_length, *extra_dims)
    """
    # Let's assume that each element in "batch" is a tuple (data, label).
    # Sort the batch in the descending order
    sorted_batch = sorted(batch, key=lambda x: x[0].shape[0], reverse=True)
    # Get each sequence and pad it
    sequences = [x[0] for x in sorted_batch]
    sequences_padded = pad_sequence(sequences, batch_first=True)
    # Also need to store the length of each sequence
    # This is later needed in order to unpad the sequences
    lengths = torch.LongTensor([len(x) for x in sequences])
    # Don't forget to grab the labels of the *sorted* batch
    labels = torch.LongTensor(list(map(lambda x: x[1], sorted_batch)))
    return sequences_padded, lengths, labels


def random_sample_sequence(batch, samples_per_batch, lengths=None, 
        out_length=None):
    """ Randomly sample from sequence data

    Parameters
    ----------
    batch : Torch.Tensor
        The batch of data to sample, with shape (batch_size, max_length, 
        *extra_dims). Data are sampled along the first two dimensions - 
        subsequent dimensions are preserved. 
    samples_per_batch
        The output shape of the sampled data will be (batch_size, 
        samples_per_batch, out_length, *extra_dims).
    lengths : torch.LongTensor, optional
        Tensor corresponding to the lengths of each sequence in the batch. These
        values are used to ensure that sampling only occurs within the valid
        region of each sequence. By default, this is set to None, indicating
        that all sequences are the same length.
    out_length : int, optional
        The output length of the sampled data, max_length by default or if
        set to None.
    """
    if lengths is not None and (torch.any(lengths < 1)):
        raise ValueError("lengths must be positive integer values.")
    batch_size = batch.shape[0]
    max_length = batch.shape[1]
    out_length = out_length if out_length is not None else max_length
    nsamples = batch_size*out_length*samples_per_batch
    batch_idx = torch.randint(low=0, high=batch_size, size=(nsamples, ))
    if lengths is not None:
        seq_idx = (torch.rand((nsamples, ))*lengths[batch_idx]).type(torch.long)
    else:
        seq_idx = torch.randint(max_length, (nsamples, ))
    out = batch.view(-1,*batch.shape[2:])[batch_idx*max_length + seq_idx]
    return out.reshape(batch_size, samples_per_batch, out_length, *batch.shape[2:])
    

def random_sample(batch, shape):
    """ Randomly sample from batched data

    Parameters
    ----------
    batch : Torch.Tensor
        The batch of data to sample, with shape (batch_size, *extra_dims). Data 
        are sampled along the batch (first) dimension - subsequent dimensions 
        are preserved. 
    shape : iterable
        The output shape of the sampled data will be (*shape, *extra_dims).

    Returns
    -------
    torch.Tensor
        Data randomly sampled from batch with shape (*shape, *extra_dims).
    """
    num_batch = batch.shape[0]
    idx = torch.randint(0, num_batch, shape)
    return batch[idx]


def negative_sample(batch):
    """ Negatively sample from batched data

    Parameters
    ----------
    batch : Torch.Tensor
        The batch of data to sample with shape (batch_size, *extra_dims). Data
        are sampled along the batch (first) dimension - subsequent dimensions
        are preserved.

    Returns
    -------
    torch.Tensor
        Output data tensor negatively sampled from batch with shape 
        (batch_size, batch_size - 1, *extra_dims). Each item contains all
        data from batch with a non-matching index.
    """
    num_batch = batch.shape[0]
    idx = torch.arange(1, num_batch).expand(num_batch, -1) + \
        torch.arange(num_batch).view(-1, 1)
    idx = torch.remainder(idx, num_batch)
    return batch[idx]
