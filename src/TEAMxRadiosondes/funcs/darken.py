import numpy as np
from matplotlib.colors import to_rgb

def darken(color, factor=0.4):
    """
    Darken an RGB or hex color by scaling it toward black.

    Parameters
    ----------
    color : str or tuple
        Matplotlib-compatible color specification (e.g. '#4477AA' or (r,g,b)).
    factor : float, optional
        Amount to darken the color.
        A value of 0 leaves the color unchanged,
        while values closer to 1 move the color toward pure black.
        Default is 0.4.

    Returns
    -------
    tuple
        A darkened RGB color tuple with values in the range [0, 1].
    """
    rgb = np.array(to_rgb(color))
    return tuple(rgb * (1 - factor))