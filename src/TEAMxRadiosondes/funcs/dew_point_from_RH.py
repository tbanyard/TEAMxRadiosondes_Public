import numpy as np
from TEAMxRadiosondes.funcs import saturation_vapour_pressure_AERK
import sys

def dew_point_from_RH(T_C, RH, p_hPa, method="vaisala"):
    """
    Calculate dew point using a variety of methods.
    
    Parameters:
        T_C: float or ndarray
            Air temperature in °C
        RH: float or ndarray
            Relative humidity in %
        p_hPa: float or ndarray
            Air pressure in hPa
        method: string
            Method to use for dewpoint calculation
            Choose from vaisala, aerk or sa90
            
    Returns:
        Td_C: float or ndarray
            Dew point temperature in °C
        description: string
            Description of the method used
    """

    # Normalise method name
    method = method.lower().strip()
    if method == "default":
        method = "vaisala"

    if method == "aerk":
        description = "Computed using the saturation vapour‐pressure expansion from Alduchov, O. A. and Eskridge, R. E. (1996) (J. Appl. Meteor., 1996, 35 (4), 601–609)"
        e_s = saturation_vapour_pressure_AERK(T_C, p_hPa)
        e = RH / 100.0 * e_s

        # Inverse AERK/AERKI for Td
        Td_C = np.where(
            T_C >= 0,
            243.04 * np.log(e / 6.1094) / (17.6251 - np.log(e / 6.1094)),
            273.86 * np.log(e / 6.1121) / (22.587 - np.log(e / 6.1121))
        )

    elif method == "vaisala":
        description = "Computed using Vaisala Magnus-type approximation (B210973EN-F)"
        A = 6.116441   # hPa
        m = 7.591386   # dimensionless
        Tn = 240.7263  # °C

        e_s = A * 10 ** (m * T_C / (T_C + Tn))
        e = RH / 100.0 * e_s

        # Compute dewpoint temperature (°C)
        Td_C = (Tn * np.log10(e / A)) / (m - np.log10(e / A))

    elif method == "sa90":
        description = "Computed using the Sonntag, D. (1990) ITS-90-consistent vapour-pressure formulation (Z. Meteor., 40, 340–344)."
        a = 17.62
        b = 243.12

        gamma = (a * T_C) / (b + T_C) + np.log(RH / 100.0)
        Td_C = (b * gamma) / (a - gamma)

    else:
        print("Dew point Method Invalid")
        sys.exit(1)

    return Td_C, description