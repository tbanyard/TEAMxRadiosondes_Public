import xarray as xr
import numpy as np

def fixnanswithmean(z):
    """
    Replace NaN values in an array with the mean of the non-NaN values.

    Parameters
    ----------
    z : array-like or xarray.DataArray
        Input data containing possible NaN values.

    Returns
    -------
    same type as input
        Data with NaN values replaced by the mean of the valid (non-NaN) elements.

    Notes
    -----
    The mean is computed while ignoring NaN values.
    """
    if hasattr(z, "fillna"):
        mean = z.mean(skipna=True)
        return z.fillna(mean)
    else:
        mean = np.nanmean(z)
        return np.where(np.isnan(z), mean, z)

