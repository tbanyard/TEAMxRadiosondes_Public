import numpy as np
from metpy.calc import moist_lapse
from metpy.units import units

def moist_adiabat_metpy(Tstart, Pstart):
    """
    Compute a moist adiabat temperature profile using MetPy.

    Parameters
    ----------
    Tstart : float
        Starting temperature (°C)
    Pstart : float
        Starting pressure (hPa)

    Returns
    -------
    Tmoist : np.ndarray
        Temperature profile along the moist adiabat (°C)
    Pmoist : np.ndarray
        Pressure levels (hPa), descending in 2 hPa steps
    """
    # Pressure levels (descending, same step as original)
    Pmoist = np.arange(Pstart, 9, -2.0) * units.hPa

    # Call MetPy moist_lapse
    Tmoist = moist_lapse(Pmoist, Tstart * units.degC).to('degC').magnitude

    # Return the same as original function
    Pmoist = Pmoist.magnitude  # strip units
    return Tmoist, Pmoist