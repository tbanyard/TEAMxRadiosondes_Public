import numpy as np

def saturation_vapour_pressure_AERK(T_C, p_hPa):
    """
    Calculate saturation vapour pressure using AERK/AERKI formulas.
    
    Parameters:
        T_C: float or ndarray
            Air temperature in °C
        p_hPa: float or ndarray
            Air pressure in hPa
            
    Returns:
        e_hPa: float or ndarray
            Saturation vapor pressure in hPa
    """
    T = np.array(T_C, dtype=float)
    p = np.array(p_hPa, dtype=float)

    e_s = np.where(
        T >= 0,
        6.1094 * np.exp((17.6251 * T) / (243.04 + T)),  # AERK over water
        6.1121 * np.exp((22.587 * T) / (273.86 + T))   # AERKI over ice
    )

    # Enhancement factor for moist air
    enhancement = np.where(
        T >= 0,
        1.00071 * np.exp(0.0000045 * p),
        0.99882 * np.exp(0.000008 * p)
    )

    return e_s * enhancement