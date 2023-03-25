import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BaseModel, Setting, OUTPUT_ACTIVATIONS


def sparsify(tensor: torch.Tensor, density: float):
    """
    Makes a tensor 'sparse' by randomly setting values to 0, such that
    the number of non-zero elements / the tensor size ~= density.

    Parameters
    ----------
    tensor
        The input tensor
    density
        The desired density

    Returns
    -------
    tensor

    """
    if density is not None:
        return tensor * (torch.rand_like(tensor) <= density).type(tensor.dtype)
    else:
        return tensor


class RandomLinear(nn.Module):
    """ Applies a linear transformation to the incoming data y = x*A^t + b

    This module is similar to torch.nn.Linear, but has different initialization
    rules and parameters. It is also usually used without the gradient.
    """
    def __init__(self, in_features, out_features, bias=True, weight_scale=1.0,
                 bias_scale=1.0, density=None, spectral_radius=None,
                 requires_grad=False):
        """

        Parameters
        ----------
        in_features
            The size of each input sample.
        out_features
            The size of each output sample.
        bias
            If set to True, adds a bias term. If set to False (default), there
            will be no additive bias.
        weight_scale
            Weight values are sampled from a uniform distribution in the range
            [-weight_scale, weight_scale].
        bias_scale
            Bias values are sampled from a uniform distribution in the range
            [-bias_scale, bias_scale].
        density
            The density of the weight matrix, i.e. the fraction of non-zero
            elements.
        spectral_radius
            If set, the weight matrix will be rescaled so that its spectral
            radius is equal to spectral_radius. For a square matrix, the
            spectral radius is the absolute value of the largest eigenvalue.
            In the case of a non-square weight matrix, the largest singular
            value is used.
        requires_grad
            Boolean if gradient information is required (for optimization).
            Default is False
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        w = torch.rand((out_features, in_features)) * 2.0 * weight_scale - weight_scale
        w = sparsify(w, density)

        if spectral_radius is not None:
            # The spectral radius is really only defined for
            # square matrices
            if in_features == out_features:
                v = w.eig()[0]
                vmag = (v * v).sum(1).sqrt()
                w /= vmag.max()
            else:
                usv = w.svd(compute_uv=False)
                w /= usv.S[0].abs()
            w *= spectral_radius

        self.weight = nn.Parameter(
            w,
            requires_grad=requires_grad
        )

        if bias:
            b = torch.rand(out_features) * 2.0 * bias_scale - bias_scale
            self.bias = nn.Parameter(sparsify(b, density),
                                     requires_grad=requires_grad)
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        if not len(x.shape):
            x = x.expand(1, 1)
        return F.linear(x, self.weight, self.bias)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


class ESNCell(BaseModel):
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
    def __init__(self, in_features, out_features, scale_in=1.0,
                 bias_scale=None, density=0.5, spectral_radius=0.9,
                 leak_rate=1.0):
        """

        Parameters
        ----------
        in_features
            The size of each input sample
        out_features
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


        References
        ----------
        [1] Lukoševičius, Mantas. "A practical guide to applying echo state
            networks." Neural networks: Tricks of the trade. Springer, Berlin,
            Heidelberg, 2012. 659-686.
        """
        super().__init__()
        bias_scale = scale_in if bias_scale is None else bias_scale
        self.in_features = Setting(in_features)
        self.out_features = Setting(out_features)
        self.scale_in = Setting(scale_in)
        self.bias_scale = Setting(bias_scale)
        self.density = Setting(density)
        self.spectral_radius = Setting(spectral_radius)
        self.leak_rate = Setting(leak_rate)
        self.configure()
    
    def configure(self):
        self.Win = RandomLinear(self.in_features, 
                                self.out_features,
                                weight_scale=self.scale_in, 
                                bias_scale=self.bias_scale,
                                requires_grad=False)
        self.Wres = RandomLinear(self.out_features, 
                                 self.out_features,
                                 weight_scale=self.scale_in,
                                 bias=False,
                                 density=self.density,
                                 spectral_radius=self.spectral_radius,
                                 requires_grad=False)
        
    def forward(self, u_in, initial_state=None):
        """ Forward pass of the ESN
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape (batch_size, num_steps, in_features)
        initial_state : torch.Tensor
            Optional input tensor with shape (batch_size, reservoir_size) to use 
            for the initial reservoir hidden state for each element in the 
            batch. If not provided, defaults to zero.
        """
        batch_size = u_in.shape[0]
        num_steps = u_in.shape[1]
        output = torch.zeros(batch_size, num_steps, self.out_features, device=u_in.device)

        # Compute the first step separately to account for initialization.
        tmp = self.Win(u_in[:,0,:])
        if initial_state is not None:
            tmp += self.Wres(initial_state)
        h_i = torch.tanh(tmp)
        output[:, 0, :] = h_i

        # Now step through the sequence.
        alpha = self.leak_rate
        beta = 1 - alpha
        for i in range(1, num_steps):
            h_i0 = torch.tanh(self.Win(u_in[:,i,:]) + self.Wres(h_i))
            h_i = alpha*h_i0 + beta*h_i
            output[:, i, :] = h_i

        return output, h_i


class ESN(BaseModel):
    """ Echo state network with linear readout """
    def __init__(self, in_features, out_features=None, reservoir_size=100, out_activation=None, **kwargs):
        """

        Parameters
        ----------
        in_features
            Dimensionality of each input sample
        out_features : int (optional)
            Dimensionality of each output sample. The default behavior, or if
            set to None, sets out_features = in_features.
        reservoir_size
            Number of nodes in the internal ESNCell. This defaults to 100
            nodes. If the ESN performs poorly, try increasing this number.
            See ESNCell for more information on this parameter.
        out_activation : str, optional
            Type of output activation. Set to None to disable. Supported options are
            "relu", "sigmoid", "tanh"
        **kwargs
            These will be passed to the internal ESNCell
        
        """
        out_features = in_features if out_features is None else out_features
        super().__init__(in_features, out_features)
        self.reservoir_size = Setting(reservoir_size)
        self.cell_parameters = Setting(kwargs)
        self.last_hidden_state = Setting(None)
        self.out_activation = Setting(out_activation)
        self.configure()

    @classmethod
    def from_checkpoint(cls, checkpoint):
        cfg = checkpoint["config"]
        out = cls(cfg["in_features"], 
                  cfg["out_features"], 
                  cfg["reservoir_size"],
                  cfg.get("out_activation", None),
                  **cfg["cell_parameters"])
        out.last_hidden_state = cfg["last_hidden_state"]
        out.load_state_dict(checkpoint["state_dict"])
        return out
    
    @staticmethod
    def description():
        return "Echo state network with linear encoder output."

    def configure(self):
        self.res = ESNCell(self.in_features, self.reservoir_size, **self.cell_parameters)
        self.lin_out = nn.Linear(self.in_features, self.out_features)
        self.esn_out = nn.Linear(self.reservoir_size, self.out_features, bias=False)
        self.out_fun = OUTPUT_ACTIVATIONS[self.out_activation]

    def forward(self, x, use_last_hidden_state=False):
        """ Forward pass of the ESN
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape (batch_size, num_steps, in_features)
        use_last_hidden_state : bool
            If True, then the last hidden state of the reservoir will be 
            used to seed the first iteration. 
        """
        h_i = self.last_hidden_state if use_last_hidden_state else None
        x_esn, self.last_hidden_state = self.res(x, h_i)
        out = self.lin_out(x) + self.esn_out(x_esn)
        return self.out_fun(out)
    