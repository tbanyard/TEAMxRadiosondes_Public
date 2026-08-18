import numpy as np

def moist_adiabat(Tstart, Pstart):
    """
    Compute a moist adiabat temperature profile using Bolton (1980).

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
        Pressure levels (hPa), descending in 10 hPa steps

    References
    ----------
    Bolton, D. (1980): "The Computation of Equivalent Potential Temperature"
    Monthly Weather Review, 108(7), 1046–1053.
    """
    # Step pressure in increments of 10 hPa
    Pmoist = np.arange(Pstart, 9, -2.0)  # hPa

    # Initialize temperature array
    Tmoist = np.zeros_like(Pmoist)
    Tmoist[0] = Tstart + 273.15  # convert to K

    # Constants
    Lv = 2.5e6   # latent heat of vaporization (J/kg)
    cp = 1004.0  # specific heat of dry air (J/kg/K)
    Rv = 461.0   # gas constant for water vapor (J/kg/K)
    Rd = 287.0   # gas constant for dry air (J/kg/K)
    g = 9.81     # gravity (m/s²)

    for i in range(1, len(Pmoist)):
        Tprev = Tmoist[i - 1]
        P = Pmoist[i]

        # Saturation vapor pressure (Bolton, Eq. 10):
        # es = 6.112 * exp(17.67 * T / (T + 243.5)) where T(°C)
        es = 6.112 * np.exp(17.67 * (Tprev - 273.15) / (Tprev - 29.65))  # hPa

        # Saturation mixing ratio (kg/kg):
        # ws = (Rd / Rv) * es / (p - es)
        #    = 0.622 * es / (p - es)
        Pmid = 0.5 * (P + Pmoist[i - 1])
        ws = 0.622 * es / (Pmid - es)  # kg/kg

        # Moist adiabatic lapse rate (Γ_m) (K/m):
        Gamma_m = (g / cp) * (1 + (Lv * ws) / (Rd * Tprev)) / (
            1 + (Lv**2 * ws) / (cp * Rv * Tprev**2)
        )

        # Integrate upward using hydrostatic approximation (Δz ≈ Δp / (ρ g))
        # ρ ≈ 1 kg/m³ -> ΔT = Γ_m * Δz = Γ_m * (Δp * 100 / g)
        #Tmoist[i] = Tprev - Gamma_m * ((Pmoist[i - 1] - P) * 100.0 / g)

        # Integration block attempt 2
        """Tv = Tprev * (1 + 0.61 * ws)  # virtual temperature
        dp = (P - Pmoist[i - 1]) * 100.0  # Pa (negative upward)
        dT = Gamma_m * (Rd * Tv / (g * P * 100.0)) * dp
        Tmoist[i] = Tprev + dT"""

        # Integration block attempt 3
        P_prev = Pmoist[i - 1] * 100.0  # Pa
        P_curr = Pmoist[i] * 100.0      # Pa
        dp = P_curr - P_prev  # negative
        P_mid = 0.5 * (P_prev + P_curr)
        Tv = Tprev * (1 + 0.61 * ws)
        dT = Gamma_m * (Rd * Tv / (g * P_mid)) * dp
        Tmoist[i] = Tprev + dT

        # Integration block attempt 4
        """dT_dP = (Rd * Tprev + Lv * ws) / (cp + (Lv**2 * ws) / (Rv * Tprev**2)) / Pmid
        dp = P - Pmoist[i - 1]
        Tmoist[i] = Tprev + dT_dP * dp"""

    # Back to °C
    Tmoist = Tmoist - 273.15

    return Tmoist, Pmoist
