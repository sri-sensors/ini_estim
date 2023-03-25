import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm


class BasicNet(nn.Module):
    def __init__(self, xdims=1, zdims=1):
        super().__init__()
        Nint = 5*(xdims + zdims)
        self.hx = nn.Linear(xdims, Nint)
        self.hz = nn.Linear(zdims, Nint)
        self.hout = nn.Linear(Nint, 1)

    def forward(self, x, z):
        y = self.hx(x) + self.hz(z)
        y = torch.relu(y)
        y = self.hout(y)
        return y


def mine_loss(model, xdata, zdata, use_bits=False):
    """ Returns the MI estimated from the NN model """
    if isinstance(xdata, torch.Tensor):
        xt = xdata
    else:
        xt = torch.from_numpy(xdata).type(torch.FloatTensor)

    if isinstance(zdata, torch.Tensor):
        zt = zdata
    else:
        zt = torch.from_numpy(zdata).type(torch.FloatTensor)
    Nsamples = zt.shape[0]
    idx = np.arange(Nsamples)
    randinds = np.random.choice(idx, Nsamples, replace=False)
    randinds[randinds==idx] = (randinds[randinds==idx] + 1) % Nsamples
    zs = zt[randinds]
    T_xz = model(xt, zt)
    T_x_z = model(xt, zs)
    v = torch.mean(T_xz) - torch.log(torch.mean(torch.exp(T_x_z)))
    if use_bits:
        v *= np.log2(np.e)
    return -v


def mi_mine(x: np.ndarray, y: np.ndarray, model: nn.Module=None,
            epochs: int=500, lr: float=0.05, return_model: bool=True,
            tqdm_callback: tqdm.tqdm = None):
    """
    Estimates MI by training a neural network. The main advantage to
    this method over other non-parametric methods (like KSG) is that
    the returned network has gradient information that can be helpful
    in running a PyTorch optimization routine.

    Parameters
    ----------
    x : np.ndarray (n_samples, n_dimensions)
        Samples of input continuous random variables
    y : np.ndarray (n_samples, m_dimensions)
        Samples of output continuous random variables
    model : nn.Module
        PyTorch network for training. Default is SimpleNet
    epochs : int
        Number of training steps (default = 500)
    lr : float
        Learning rate (default = 0.05)
    return_model
        Set to True to return the trained model and loss history (default).
    tqdm_callback
        A tqdm instance to update during learning. Default is None

    Returns
    -------
    mi : float
        The estimated mutual information

    if return_model, returns also:
    model : nn.Module
        The model that estimated the mutual information
    loss_history
        The estimated mutual information at each training epoch.

    Notes
    -----
    The returned model is only valid for estimating MI under similar conditions
    as the training was performed. If the conditions change, then it is better
    to re-train the model.

    References
    ----------
    [1] Belghazi, Mohamed Ishmael, et al. "Mine: mutual information neural
        estimation." arXiv preprint arXiv:1801.04062 (2018).
    """
    if x.ndim < 2:
        x = x.reshape((-1, 1))
    if y.ndim < 2:
        y = y.reshape((-1, 1))

    xt = torch.from_numpy(x).type(torch.FloatTensor)
    yt = torch.from_numpy(y).type(torch.FloatTensor)
    model, MIest = train_mine_model(xt, yt, model, epochs, lr, tqdm_callback)
    mi_out = -mine_loss(model, x, y).item()
    # convert to bits.
    mi_out *= np.log2(np.e)

    if return_model:
        return mi_out, model, MIest
    else:
        return mi_out


def train_mine_model(xdata, ydata, model=None, epochs: int = 500, lr: float = 0.05,
                     tqdm_callback: tqdm.tqdm = None):
    """

    Parameters
    ----------
    xdata
    ydata
    model
    epochs
    lr
    tqdm_callback

    Returns
    -------
    model, mutual info history

    """
    model = model if model is not None else BasicNet(xdata.shape[1], ydata.shape[1])
    opt = optim.Adam(model.parameters(), lr=lr)

    Nsamples = xdata.shape[0]
    bs = Nsamples // 10
    indexes = np.arange(Nsamples)
    MIest = []

    if tqdm_callback is not None:
        tqdm_callback.total = epochs

    for epoch in range(epochs):
        subinds = np.random.choice(indexes, bs, replace=False)
        loss = mine_loss(model, xdata[subinds], ydata[subinds])
        mi_val = -loss.data.numpy()
        MIest.append(mi_val)
        model.zero_grad()
        loss.backward()
        opt.step()
        if tqdm_callback is not None:
            tqdm_callback.update()

    return model, MIest
