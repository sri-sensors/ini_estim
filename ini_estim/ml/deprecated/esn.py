import torch
from torch import nn as nn
from ini_estim.ml.models.esn import RandomLinear


def _fix_dims(x, num_features):
    if not len(x.shape):
        x = x.expand(1, -1)
    if len(x.shape) == 1:
        if num_features == 1:
            x = x[:, None]
        else:
            x = x.expand(1, -1)
    return x


class ESNReservoir(nn.Module):
    """ Module implementing the reservoir of an Echo State Network

    Inspired by: https://github.com/danieleds/Reservoir-Computing-in-PyTorch

    References
    ----------
    [1] Lukoševičius, Mantas. "A practical guide to applying echo state
        networks." Neural networks: Tricks of the trade. Springer, Berlin,
        Heidelberg, 2012. 659-686.
    [2] LUKOŠEVIČIUS, Mantas; JAEGER, Herbert. Reservoir computing approaches
        to recurrent neural network training. Computer Science Review, 2009,
        3.3: 127-149.
    """
    def __init__(self, in_features, reservoir_size, scale_in=1.0,
                 bias_scale=None, density=0.5, spectral_radius=0.9,
                 leak_rate=1.0, f=torch.tanh):
        """

        Parameters
        ----------
        in_features
            The size of each input sample
        reservoir_size
            The number of nodes in the reservoir. [1] recommends starting with
            a small reservoir to tune the other global parameters, and then
            scale to a larger reservoir. In general, the more complicated the
            task, the larger the reservoir should be.
        scale_in
            The scale of the input data weights, drawn from a uniform
            distribution [-scale_in, scale_in]. This value is data dependent.
            See [1] for more guidance. Defaults to 1.0.
        bias_scale
            The scale of the input bias weights, drawn from a uniform
            distribution [-bias_scale, bias_scale]. If set to None (default),
            bias_scale will be set to scale_in. [1] observes that performance
            can often be improved by setting bias_scale != scale_in.
        density
            The fraction of non-zero reservoir weights. Defaults to 0.5. [1]
            recommends starting with ~10 connections per reservoir node, and
            also observes that this value does not significantly affect the
            reservoir's effectiveness - making it a low optimization priority.
        spectral_radius
            The spectral radius of the reservoir weight matrix. This value must
            be low enough to satisfy the "Echo State Network" condition. That
            usually means spectral_radius < 1.0, but according to [1], is not
            necessarily required. The spectral_radius is also seen as the
            decay rate of the reservoir's memory - so this value should be
            generally higher in cases where more memory is required.
        leak_rate
            The leak rate of the reservoir's integration action. If set to 1.0
            (default), there is no integration.
            xh[i] = f(Win(u[i]) + Wres(x[i-1]))
            x[i] = (1-leak_rate)*x[i-1] + leak_rate*xh[i]
        f
            The internal activation function. Default is torch.tanh.

        References
        ----------
        [1] Lukoševičius, Mantas. "A practical guide to applying echo state
            networks." Neural networks: Tricks of the trade. Springer, Berlin,
            Heidelberg, 2012. 659-686.
        """
        super().__init__()
        bias_scale = scale_in if bias_scale is None else bias_scale
        self.in_features = in_features
        self.reservoir_size = reservoir_size
        self.f = f
        self.Win = RandomLinear(in_features,
                                reservoir_size,
                                weight_scale=scale_in,
                                bias_scale=bias_scale,
                                requires_grad=False)
        self.Wres = RandomLinear(reservoir_size, reservoir_size,
                                 weight_scale=scale_in,
                                 bias=False,
                                 density=density,
                                 spectral_radius=spectral_radius,
                                 requires_grad=False)
        self.leak_rate = leak_rate
        self.last_state = None

    def reset_state(self):
        self.last_state = None

    def forward(self, u_in, initial_state=None):
        if initial_state is None:
            initial_state = self.last_state

        num_steps = u_in.shape[0]
        x = torch.zeros((num_steps, self.reservoir_size))

        if initial_state is not None:
            xi = self.f(self.Win(u_in[0]) + self.Wres(initial_state))
            x[0] = (1 - self.leak_rate) * initial_state + self.leak_rate*xi
        else:
            x[0] = self.f(self.Win(u_in[0]))

        for i in range(1, num_steps):
            xi = self.f(self.Win(u_in[i]) + self.Wres(x[i - 1]))
            x[i] = (1 - self.leak_rate) * x[i - 1] + self.leak_rate * xi

        self.last_state = x[-1]
        return x

    def extra_repr(self):
        return 'in_features={}, reservoir_size={}, leak_rate={}'.format(
            self.in_features, self.reservoir_size, self.leak_rate
        )


class ESNLinear(nn.Module):
    """ Echo state network with linear readout """
    def __init__(self, in_features, out_features=None, reservoir_size=100,
                 **kwargs):
        """
        Parameters
        ----------
        in_features
            Dimensionality of each input sample
        out_features : int (optional)
            Dimensionality of each output sample. The default behavior, or if
            set to None, sets out_features = in_features.
        reservoir_size
            Number of nodes in the internal ESNReservoir. This defaults to 100
            nodes. If the ESN performs poorly, try increasing this number.
            See ESNReservoir for more information on this parameter.
        **kwargs
            These will be passed to the internal ESNReservoir
        """
        super().__init__()

        out_features = in_features if out_features is None else out_features
        Nint = reservoir_size + in_features

        self.out_features = out_features
        self.res = ESNReservoir(in_features, reservoir_size, **kwargs)
        self.Wout = nn.Linear(Nint, out_features)
        self.last_output = None
        self.register_buffer('YtX', torch.zeros((out_features, Nint + 1),
                                                requires_grad=False))
        self.register_buffer('XtX', torch.zeros((Nint + 1, Nint + 1),
                                                requires_grad=False))

    def reset_regression(self):
        """ Reset the internal regression parameter buffers """
        self.YtX.zero_()
        self.XtX.zero_()

    def reset_state(self):
        """ Re-initialize the state of the ESN

        Zeros out ESN reservoir state and last readout output state. Does NOT
        reset readout weights.
        """
        self.last_output = None
        self.res.reset_state()

    def forward(self, x, y_train=None, lam=1e-3):
        """ compute the ESN with linear readout

        Parameters
        ----------
        x : torch.Tensor (# steps, # input features)
            input to the ESN. If set to None, the ESN will attempt to use its
            last output as the input.
        y_train : (optional) torch.Tensor (#steps, # output features)
            Desired output of the ESN. Supplying these values will
            update the internal ridge regression parameter buffers
            and re-compute the linear output layer using the parameters.
            Only supply this if you are doing ridge regression.
        lam : (optional) scalar
            Regularization parameter for ridge regression
        """
        if x is None:
            x = self.last_output

        x = _fix_dims(x, self.res.in_features)
        out = self.res(x)
        if y_train is not None:
            # training with regression
            y_train = _fix_dims(y_train, self.out_features)
            X = torch.cat((torch.ones(x.shape[0], 1), x, out), 1)
            self.YtX += torch.mm(y_train.t(), X)
            self.XtX += torch.mm(X.t(), X)
            self.finish_regression(lam, reset_training=False)

        xall = torch.cat((x, out), dim=1)
        yout = self.Wout(xall)
        self.last_output = yout[-1]
        return yout

    def finish_regression(self, lam=1e-3, reset_training=True):
        """ Solve for linear readout weights

        Requires that the model is run with training data first.

        Parameters
        ----------
        lam : float
            ridge regression parameter

        """
        # Original (works OK with numpy version of routines, but PyTorch
        # gives different answers. This may just be a numerical conditioning
        # thing).
        # A = torch.pinverse(self.XtX + lam * torch.eye(self.XtX.shape[0]))
        # W = torch.mm(self.YtX, A)
        #
        # pytorch's solve seems to have better numerical behavior than
        # using pinverse + mm.
        W, _ = torch.solve(
            self.YtX.t(), self.XtX + lam * torch.eye(self.XtX.shape[0])
        )
        W = W.t()
        self.Wout.bias.data = W[:, 0].clone()
        self.Wout.weight.data = W[:, 1:].clone()
        self.Wout.zero_grad()
        if reset_training:
            self.reset_regression()
