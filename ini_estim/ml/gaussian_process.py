import numpy as np


def rbf_kernel(x1, x2, variance=1.0):
    """ Radial basis function kernel.

    Basically, gaussian distance. A similarity measure, since the value of the
    RBF kernel decreases with distance and ranges between zero (in the limit)
    and one (when x1 = x2)

    Parameters
    ----------
    x1, x2 : np.ndarray
        1-d or scalar values
    variance : scalar
        Free parameter. Higher values increase smoothness
    """
    return np.exp(-0.5 * ((x1 - x2) ** 2) / variance)


def sigmoid_kernel(x1, x2, gamma=1.0, coef0=1.0):
    """The sigmoid kernel (hyperbolic tangent)

    Often used as neuron activation function in neural networks (multi-layer
    perceptrons). NOT a true kernel because it doesn't produce positive
    semi-definite covariance matrices.

    Parameters
    ----------
    x1, x2
        Feature vectors on some input space

    """

    return np.tanh(gamma * x1 * x2 + coef0)


def gram_matrix(xs, kernel_fun=rbf_kernel, **kwargs):
    """ Compute the kernel function for all sample pairs

    xs : np.ndarray
        1-D array of sample locations
    kernel_fun : function
        This is a kernel function in the form kernel_fun(x1, x2, **kwargs).
        The default kernel function is 'rbf_kernel'
    **kwargs
        These arguments will be passed to the kernel function
    """
    return kernel_fun(xs[None, :], xs[:,None], **kwargs)


def generate_gp(num_samples, kernel_fun=rbf_kernel, generator=np.random.default_rng(), **kwargs):
    """ Generate a 0-mean signal from a Gaussian process

    Parameters
    ----------
    num_samples
        Number of samples to generate
    kernel_fun : function
        This is a kernel function in the form kernel_fun(x1, x2, **kwargs).
        The default kernel function is 'rbf_kernel'
    generator : np.random.Generator, Optional
        The random state generator, np.random.default_rng() by default.
    **kwargs
        These arguments will be passed to the kernel function
    """
    xs = np.arange(num_samples) - num_samples // 2
    C = gram_matrix(xs, kernel_fun, **kwargs)
    return generator.multivariate_normal(np.zeros(num_samples), C)

