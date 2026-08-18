import numpy as np
from matplotlib.colors import to_rgb

def lighten(color, factor=0.6):
    """
    Lighten an RGB or hex color by blending it toward white.

    Parameters
    ----------
    color : str or tuple
        Matplotlib-compatible color specification (e.g. '#4477AA' or (r,g,b)).
    factor : float, optional
        Amount to lighten the color.
        A value of 0 leaves the color unchanged,
        while values closer to 1 move the color toward pure white.
        Default is 0.6.

    Returns
    -------
    tuple
        A lightened RGB color tuple with values in the range [0, 1].
    """

    rgb = np.array(to_rgb(color))
    return tuple(1 - (1 - rgb) * (1 - factor))