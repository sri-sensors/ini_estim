import numpy as np


def random_points_circle(num_pts, diameter=1.0):
    """
    Generate a random distribution of points in a circle
    http://mathworld.wolfram.com/DiskPointPicking.html

    Parameters
    ----------
    num_pts
        Number of points to generate
    diameter
        Diameter of circle

    Returns
    -------
    x, y
        Randomly distributed points within circle

    """
    r = np.random.rand(num_pts)
    theta = np.random.rand(num_pts)*2*np.pi
    radius = diameter*0.5
    x = radius*np.sqrt(r)*np.cos(theta)
    y = radius*np.sqrt(r)*np.sin(theta)
    return x, y


def uniform_points_circle(num_pts, diameter=1.0):
    """
    Pick uniformly distributed points within circle.
    Uses "Vogel's Method" from:
    https://www.arl.army.mil/arlreports/2015/ARL-TR-7333.pdf

    Parameters
    ----------
    num_pts
    diameter

    Returns
    -------

    """
    radius = diameter*0.5
    n = np.arange(0, num_pts)
    r = radius*np.sqrt((n+1)/num_pts)
    theta = np.pi*(3-np.sqrt(5))*n
    x = r*np.cos(theta)
    y = r*np.sin(theta)
    return x, y


def random_points_rectangle(num_pts, x_size, y_size):
    """Generate a random distribution of points on a rectangle centered on
    (0, 0)."""
    x = np.random.rand(num_pts)*x_size - x_size/2
    y = np.random.rand(num_pts)*y_size - y_size/2
    return x, y


def grid_points_rectangle(num_pts, x_size, y_size, min_sep=0):
    """Generate a gridded distribution of points on a rectangle centered on
    (0, 0)."""

    # distance between points
    d = np.sqrt(x_size*y_size/num_pts)

    if d < min_sep:
        raise ValueError(
            "Number of points is too high for specified minimum separation")
    xx = np.arange(-x_size/2, x_size/2, d)
    yy = np.arange(-y_size/2, y_size/2, d)
    x, y = np.meshgrid(xx, yy)
    x[::2, :] = x[::2, :] + d/2
    return x, y
