import numpy as np

def dry_adiabat(Tstart, Pstart):
    """
    Compute a dry adiabat temperature profile.

    Parameters
    ----------
    Tstart : float
        Starting temperature (°C)
    Pstart : float
        Starting pressure (hPa)

    Returns
    -------
    Tdry : np.ndarray
        Temperature profile along the dry adiabat (°C)
    Pdry : np.ndarray
        Pressure levels (hPa), descending in 10 hPa steps
    """
    # Constants
    Rd = 287.0     # J/kg/K
    cp = 1004.0    # J/kg/K
    kappa = Rd / cp

    # Pressure grid
    Pdry = np.arange(Pstart, 9, -10.0)  # from Pstart down to 10 hPa

    # Convert start temperature to Kelvin
    T0 = Tstart + 273.15

    # Potential temperature (theta)
    theta = T0 * (1000.0 / Pstart) ** kappa

    # Temperature profile along the dry adiabat
    Tdry = theta * (Pdry / 1000.0) ** kappa

    # Convert back to °C
    Tdry = Tdry - 273.15

    return Tdry, Pdry
