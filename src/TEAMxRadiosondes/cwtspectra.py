# Note: Claude and GPT have been used to enhance comments and assist with the
# creation of data parsing code. All scientific code has been thoroughly checked
# to ensure it is correct and the author takes full responsibility for the output.

import os
import sys
from TEAMxRadiosondes.funcs import haversine
import xarray as xr
import netCDF4 as nc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse
import matplotlib.dates as mdates
from matplotlib import ticker as mticker
from matplotlib.patches import FancyBboxPatch
from matplotlib.gridspec import GridSpec
from matplotlib.colors import TwoSlopeNorm
from matplotlib import cm
from scipy import signal
from scipy.ndimage import uniform_filter1d, gaussian_filter, uniform_filter
from scipy.signal import fftconvolve, butter, filtfilt
from scipy.interpolate import interp1d, RegularGridInterpolator
import time

def four_digit_formatter(x, pos):
    """Format colorbar tick as 4-digit number, max 99999"""
    x = min(x, 99999)
    return f'{int(x):5d}'

def lag1_autocorr(x):
    """Estimated lag-1 autocorrelation of a 1D array, using Torrence & Compo (1998) method."""
    x = x - np.nanmean(x)
    return np.nansum(x[:-1]*x[1:]) / np.nansum(x[:-1]**2)

def initialise_rs_dict(f, rs_dict):
    """Load a radiosonde dataset from a netCDF file and initialise
    a dictionary to store all relevant variables for this sonde."""
    ds = xr.load_dataset(f)
    ds.load()
    ds.close()

    # Setting the key for rs_dict, to handle cases where
    # there are two sondes with the same serial number
    snumber = ds.attrs['serial']
    if snumber not in rs_dict:
        key = snumber
    else:
        rs_dict[f"{snumber}_A"] = rs_dict.pop(snumber)
        key = f"{snumber}_B"

    # Initialise radiosonde dictionary
    rs_dict[key] = {
        "file": "../data/{file}".format(file=os.path.basename(f)),
        "ds": ds,
        'ds_sorted': None,
        "z": None,
        "lat": None,
        "lon": None,
        "f": None,
        "f_prime": None,
        "background": None,
        "coeffs_complex": None,
        "power": None,
    }

    return rs_dict

def run_butterworth_filter(rs_dict, config, fvar):
    """Apply a Butterworth filter to the radiosonde data to separate the
    background and perturbation components.
    --> Uses vel_h, the horizontal wind speed, as the input signal for filtering.
    --> Returns the updated rs_dict with new keys 'f_prime' and 'background'."""
    snumber = list(rs_dict.keys())[-1]

    # Rename fvar if it's 'w' to match the dataset variable name
    if fvar == 'w':
        fvar = 'vel_z_prime_smoothed'

    # Sort ds so altitude increases monotonically
    rs_dict[snumber]['ds_sorted'] = rs_dict[snumber]['ds'].sortby('alt')
    rs_dict[snumber]['z'] = rs_dict[snumber]['ds_sorted']['alt'].values
    rs_dict[snumber]['f'] = rs_dict[snumber]['ds_sorted'][fvar].values
    rs_dict[snumber]['lat'] = rs_dict[snumber]['ds_sorted']['lat'].values
    rs_dict[snumber]['lon'] = rs_dict[snumber]['ds_sorted']['lon'].values
    rs_dict[snumber]['time'] = rs_dict[snumber]['ds_sorted']['time'].values

    # Obtain dz to calculate nyquist freq. for Butterworth cutoff wavelength
    rs_dict[snumber]['dz'] = np.nanmean(np.diff(rs_dict[snumber]['z']))

    # Apply a NaN mask to remove NaNs
    mask = np.isfinite(rs_dict[snumber]['z']) & np.isfinite(rs_dict[snumber]['f']) & np.isfinite(rs_dict[snumber]['lat']) & np.isfinite(rs_dict[snumber]['lon']) & np.isfinite(rs_dict[snumber]['time'])
    rs_dict[snumber]['z'] = rs_dict[snumber]['z'][mask]
    rs_dict[snumber]['f'] = rs_dict[snumber]['f'][mask]
    rs_dict[snumber]['lat'] = rs_dict[snumber]['lat'][mask]
    rs_dict[snumber]['lon'] = rs_dict[snumber]['lon'][mask]
    rs_dict[snumber]['time'] = rs_dict[snumber]['time'][mask]

    # Butterworth filter
    cutoff_wavelength = config['BUTTERWORTH_CUTOFF_WAVELENGTH'] # Manually select cut-off wavelength in km
    cutoff_freq = 1.0 / (cutoff_wavelength * 1000) # Compute freq in SI units
    nyquist = 1.0 / (2.0 * rs_dict[snumber]['dz']) # Compute nyquist
    normal_cutoff = cutoff_freq / nyquist

    b, a = butter(config["BUTTER_ORDER"], normal_cutoff, btype='highpass')
    rs_dict[snumber]['f_prime'] = filtfilt(b, a, rs_dict[snumber]['f'])
    rs_dict[snumber]['background'] = rs_dict[snumber]['f'] - rs_dict[snumber]['f_prime']

    return rs_dict

def compute_single_spectrum(f, rs_dict, config, args):
    """Compute the Morlet CWT power spectrum for a single radiosonde.
    Note: This is currently on the original (irregular) altitude grid,
    and is not interpolated onto a regular grid beforehand."""
    snumber = list(rs_dict.keys())[-1]

    ##############################################
    ### Run Morlet CWT to obtain power spectra ###
    ##############################################

    # Morlet wavelet transform
    m0 = config["MORLET_CENTRAL_WAVENUMBER_M_0"]
    signal_profile = rs_dict[snumber]['f_prime']

    # Define vertical wavelengths required (in metres)
    wavelengths = np.linspace(1000, 15000, 100)  # 1–15 km

    # Convert wavelength to scale
    # Note: This equation is a rearranged version of the standard Morlet wavelength-scale relationship:
    # λ = (4πs) / (m0 + sqrt(2 + m0^2)), which can be found in Table 1. of Torrence & Compo (1998).
    scales = wavelengths * (m0 + np.sqrt(2 + m0**2)) / (4 * np.pi)   # λ ≈ 1.03 s for m0=6

    # Initialise power and coefficients arrays
    rs_dict[snumber]['power'] = np.zeros((len(scales), len(signal_profile)))
    rs_dict[snumber]['coeffs_complex'] = np.zeros((len(scales), len(signal_profile)), dtype=complex)

    # Admissibility correction and unit-norm constant (exact for any m0)
    # For m0<6, these may start to affect the shape of the wavelet
    kappa = np.exp(-m0**2 / 2)
    c_sigma = (1 + np.exp(-m0**2) - 2 * np.exp(-3 * m0**2 / 4))**(-0.5)
    if m0 < 6:
        print(f"Warning: m0={m0} is below 6. Admissibility correction c_sigma={c_sigma:.6f}, "
              f"kappa={kappa:.6f} are non-negligible and have been applied.")

    # Loop over wavelengths/scales to compute wavelet coefficients and power
    for i, s in enumerate(scales):

        # Define wavelet in height coordinates
        eta = np.arange(-4*s, 4*s, rs_dict[snumber]['dz']) / s
        
        morlet = c_sigma * (np.pi**(-0.25)) * (np.exp(1j * m0 * eta) - kappa) * np.exp(-eta**2 / 2)

        # Normalize for scale
        morlet = morlet / np.sqrt(s)

        # Convolve
        coeff = fftconvolve(signal_profile, np.conj(morlet[::-1]), mode='same') * rs_dict[snumber]['dz']
        rs_dict[snumber]['coeffs_complex'][i, :] = coeff
        rs_dict[snumber]['power'][i, :] = np.abs(coeff)**2 / s

    #####################
    ### Plotting Code ###
    #####################

    fig, ax1 = plt.subplots(figsize=(8,8))
    plt.subplots_adjust(left=0.1, right=0.9)
    # Bottom x-axis: k_z
    k_z = 2 * np.pi / wavelengths  # rad/m
    cf = ax1.contourf(
        k_z,       # x-axis: vertical wavenumber
        rs_dict[snumber]['z']/1000.0,         # y-axis: altitude
        rs_dict[snumber]['power'].T,   # transpose so shape matches (alt, k_z)
        levels=50,
        cmap='viridis'
    )

    ax1.set_xlabel('Vertical Wavenumber $k_z$ (rad/m)')
    ax1.set_ylabel('Altitude (km)')
    ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))

    # --- Bottom axis (wavenumber) ---
    ax1.set_xscale('log')
    ax1.set_xlim(k_z.min(), k_z.max())

    # Clean ticks
    lambda_ticks_km = np.array([15, 10, 5, 3, 2, 1.5, 1])
    lambda_ticks_m = lambda_ticks_km * 1000
    #k_ticks = 2 * np.pi / lambda_ticks_m
    k_ticks = np.array([0.0005, 0.001, 0.002, 0.005])
    ax1.set_xticks(k_ticks)
    ax1.set_xticklabels([
        r"$%.1f\times10^{%d}$" % (kt/10**np.floor(np.log10(kt)), np.floor(np.log10(kt)))
        for kt in k_ticks
    ])
    #ax1.set_xticklabels([f'{kt:.1e}' for kt in k_ticks])
    ax1.set_xlabel('Vertical Wavenumber $k_z$ (rad/m)')

    # Top axis (wavelength)
    ax2 = ax1.twiny()
    ax2.set_xscale('log')
    ax2.set_xlim(ax1.get_xlim())

    k_top_ticks = 2 * np.pi / lambda_ticks_m
    ax2.set_xticks(k_top_ticks)
    ax2.set_xticklabels([f'{lt:g}' for lt in lambda_ticks_km])
    ax2.set_xlabel('Vertical Wavelength $\\lambda_z$ (km)')

    # Remove messy minor ticks
    ax1.minorticks_off()
    ax2.minorticks_off()

    cbar = plt.colorbar(cf, ax=ax1, label='Wavelet Power (m$^2$/s$^2$) ')
    #cbar.ax.yaxis.set_major_formatter(mticker.FuncFormatter(four_digit_formatter))

    # Extract launch time
    t = pd.to_datetime(rs_dict[snumber]['ds'].attrs["launch_time"])
    timestamp = t.strftime("%Y%m%d_%H%M%S")
    prettytimestamp = t.strftime("%d %b %Y %H:%M:%S UTC")

    # Build figure name
    figname = f"{timestamp}_{rs_dict[snumber]['ds'].attrs.get('serial','')}_pspectra.png"
    plt.title(
        f"CWT Power Spectrum (Morlet m0=6)\n"
        f"{rs_dict[snumber]['ds'].attrs['site_name']}: {rs_dict[snumber]['ds'].attrs['serial']} ({rs_dict[snumber]['ds'].attrs.get('source','')})\n"
        f"{t:%d %b %Y %H:%M:%S UTC}",
        fontsize = 10
    )

    # Ensure the 'plots' subdirectory exists (create if needed)
    os.makedirs("plots", exist_ok=True)
    plot_path = os.path.join("plots", figname)
    plt.savefig(plot_path, dpi=225, bbox_inches='tight', pad_inches=0.02)
    print(f"Saved figure {figname} in directory {os.path.join(os.getcwd(), 'plots')} using compute_single_spectrum")
    print("Note: These _psectra.png files are generated from the individually processed sondes, and are not interpolated onto a regular grid beforehand.")

    plt.close()

    return rs_dict

def compute_cospectra(rs_dict, config):
    ######################
    ### Cospectra Code ###
    ######################
    # Below is the dictionary which all plottable results will
    # be stored in and returned at the end of this function
    results_dict = {}
    s1, s2 = list(rs_dict.keys())[:2]

    # Use a regular altitude grid (i.e. 5 m spacing)
    z1, z2 = rs_dict[s1]['z'], rs_dict[s2]['z']
    dz = 5  # 5 m spacing
    z_reg = np.arange(max(z1[0], z2[0]), min(z1[-1], z2[-1]), dz)

    # Interpolate coefficients along z
    interp_rs1 = interp1d(z1, rs_dict[s1]['f_prime'], kind='linear', bounds_error=False, fill_value=np.nan)
    interp_rs2 = interp1d(z2, rs_dict[s2]['f_prime'], kind='linear', bounds_error=False, fill_value=np.nan)
    rs_dict[s1]['f_prime_reg'] = interp_rs1(z_reg)
    rs_dict[s2]['f_prime_reg'] = interp_rs2(z_reg)

    # Interpolate lat/lon for both sondes (for dxy calculation and plotting)
    interp_lat1 = interp1d(z1, rs_dict[s1]['lat'], bounds_error=False, fill_value=np.nan)
    interp_lon1 = interp1d(z1, rs_dict[s1]['lon'], bounds_error=False, fill_value=np.nan)
    interp_lat2 = interp1d(z2, rs_dict[s2]['lat'], bounds_error=False, fill_value=np.nan)
    interp_lon2 = interp1d(z2, rs_dict[s2]['lon'], bounds_error=False, fill_value=np.nan)
    rs_dict[s1]['lat_reg'] = interp_lat1(z_reg)
    rs_dict[s1]['lon_reg'] = interp_lon1(z_reg)
    rs_dict[s2]['lat_reg'] = interp_lat2(z_reg)
    rs_dict[s2]['lon_reg'] = interp_lon2(z_reg)

    lat1 = rs_dict[s1]['lat_reg']
    lon1 = rs_dict[s1]['lon_reg']
    lat2 = rs_dict[s2]['lat_reg']
    lon2 = rs_dict[s2]['lon_reg']
    dxy_sep_km = np.empty_like(lat1, dtype=float)
    for i in range(len(lat1)):
        dxy_sep_km[i] = haversine([lat1[i], lat2[i]], [lon1[i], lon2[i]])
    dxy_sep_m = dxy_sep_km * 1000

    t1, t2 = rs_dict[s1]['time'], rs_dict[s2]['time']
    t1 = pd.to_datetime(t1).astype('int64') / 1e9
    t2 = pd.to_datetime(t2).astype('int64') / 1e9
    t_common = np.linspace(max(t1.min(), t2.min()),
                    min(t1.max(), t2.max()),
                    500)
    z1_t = interp1d(t1, rs_dict[s1]['z'], bounds_error=False, fill_value=np.nan)(t_common)
    z2_t = interp1d(t2, rs_dict[s2]['z'], bounds_error=False, fill_value=np.nan)(t_common)
    dz_sep = z2_t - z1_t
    interp_t1 = interp1d(z1, t1, bounds_error=False, fill_value=np.nan)
    interp_t2 = interp1d(z2, t2, bounds_error=False, fill_value=np.nan)
    t1_common = interp_t1(z_reg)
    t2_common = interp_t2(z_reg)
    dt_sep = t2_common - t1_common

    # Morlet wavelet transform
    m0 = config["MORLET_CENTRAL_WAVENUMBER_M_0"]

    # Instrument precision for uncertainty calculations (in m/s)
    # WEA-MET-RS41-Performance-White-paper-B211356EN-B-LOW-v3.pdf (Chapter 6)
    instrument_precision = 0.15

    # Define vertical wavelengths required (in metres)
    wavelengths = np.linspace(1000, 15000, 100)  # 1–15 km

    # Convert wavelength to scale
    # Note: This equation is a rearranged version of the standard Morlet wavelength-scale relationship:
    # λ = (4πs) / (m0 + sqrt(2 + m0^2)), which can be found in Table 1. of Torrence & Compo (1998).
    scales = wavelengths * (m0 + np.sqrt(2 + m0**2)) / (4 * np.pi)   # λ ≈ 1.03 s for m0=6

    # Initialise power, coefficients and uncertainty arrays along new regular grid
    rs_dict[s1]['coeffs_complex_reg'] = np.zeros((len(scales), len(z_reg)), dtype=complex)
    rs_dict[s2]['coeffs_complex_reg'] = np.zeros((len(scales), len(z_reg)), dtype=complex)
    rs_dict[s1]['power_reg'] = np.zeros_like(rs_dict[s1]['coeffs_complex_reg'], dtype=float)
    rs_dict[s2]['power_reg'] = np.zeros_like(rs_dict[s2]['coeffs_complex_reg'], dtype=float)
    sigma_W_per_scale = np.zeros(len(scales))

    # Admissibility correction and unit-norm constant (exact for any m0)
    # For m0<6, these may start to affect the shape of the wavelet
    kappa = np.exp(-m0**2 / 2)
    c_sigma = (1 + np.exp(-m0**2) - 2 * np.exp(-3 * m0**2 / 4))**(-0.5)
    if m0 < 6:
        print(f"Warning: m0={m0} is below 6. Admissibility correction c_sigma={c_sigma:.6f}, "
              f"kappa={kappa:.6f} are non-negligible and have been applied.")

    # Loop over wavelengths/scales to compute wavelet coefficients and power
    for i, s in enumerate(scales):
        # Set η ∈ [-4, +4] in height coordinates, which corresponds to ±4s in actual height units.
        # Since the Morlet wavelet's Gaussian envelope is exp(-η²/2), this ensures the wavelet is
        # effectively zero at the edges, i.e. exp(-8) ≈ 0.00034, minimizing edge effects in the convolution.
        eta = np.arange(-4*s, 4*s, dz) / s

        morlet = c_sigma * (np.pi**(-0.25)) * (np.exp(1j * m0 * eta) - kappa) * np.exp(-eta**2 / 2)
        morlet = morlet / np.sqrt(s)

        coeff1 = fftconvolve(rs_dict[s1]['f_prime_reg'], np.conj(morlet[::-1]), mode='same') * dz
        coeff2 = fftconvolve(rs_dict[s2]['f_prime_reg'], np.conj(morlet[::-1]), mode='same') * dz

        rs_dict[s1]['coeffs_complex_reg'][i, :] = coeff1
        rs_dict[s2]['coeffs_complex_reg'][i, :] = coeff2

        rs_dict[s1]['power_reg'][i, :] = np.abs(coeff1)**2 / s
        rs_dict[s2]['power_reg'][i, :] = np.abs(coeff2)**2 / s

        # Numerical noise in W for uncertainty calculations
        sigma_W_per_scale[i] = np.sqrt(instrument_precision**2 * dz**2 * np.sum(np.abs(morlet)**2))

    cross_coeffs = rs_dict[s1]['coeffs_complex_reg'] * np.conj(rs_dict[s2]['coeffs_complex_reg'])
    
    # Wavelet Cospectrum [m² s⁻²]: Re{W₁ W₂*} / s
    cospectrum = np.real(cross_coeffs) / scales[:, None]
    # Wavelet Quadrature Spectrum [m² s⁻²]: Im{W₁ W₂*} / s
    quadrature = np.imag(cross_coeffs) / scales[:, None]
    # Wavelet Cross-Power [m² s⁻²]: |W₁ W₂*| / s
    cross_power = np.abs(cross_coeffs) / scales[:, None]
    # Wavelet Phase [rad]: ∠(W₁ W₂*)
    phase = np.angle(cross_coeffs)

    """
    Stokes' Parameters (valid when there is one sonde, two components u and v)
    Note: A little more research is still needed to understand the importance of these derived variables
    """

    I = (np.abs(rs_dict[s1]['coeffs_complex_reg'])**2 + np.abs(rs_dict[s2]['coeffs_complex_reg'])**2) / scales[:, None]
    D = (np.abs(rs_dict[s1]['coeffs_complex_reg'])**2 - np.abs(rs_dict[s2]['coeffs_complex_reg'])**2) / scales[:, None]
    P = 2 * np.real(cross_coeffs) / scales[:, None]
    Q = 2 * np.imag(cross_coeffs) / scales[:, None]

    eps = 1e-12 # Epsilon (machine precision) to prevent division by zero in DoP and chi calculations
    DoP   = np.sqrt(D**2 + P**2 + Q**2) / (I + eps)  # degree of polarisation
    theta = 0.5 * np.arctan2(P, D) # orientation angle of major axis
    chi   = 0.5 * np.arcsin(Q / (np.sqrt(D**2 + P**2 + Q**2) + eps)) # ellipticity angle
    
    sigma_phase = sigma_W_per_scale[:, None] * np.sqrt(2) / np.sqrt(np.abs(cross_coeffs) + eps) # σ_φ = σ_W * √2 / √(|X₁₂| + ϵ)
    sigma_phase_deg = np.degrees(sigma_phase)

    """
    Below this concerns me still... Above seems ok.
    """

    ### Computing coherence ###
    # Smoothing scales
    sigma_scale = 2      # smoothing across wavelengths
    sigma_height = 10    # smoothing across altitude points

    # Apply smoothing
    power1 = np.abs(rs_dict[s1]['coeffs_complex_reg'])**2
    power2 = np.abs(rs_dict[s2]['coeffs_complex_reg'])**2
    Sxy = gaussian_filter(cross_coeffs, sigma=(sigma_scale, sigma_height))
    Sxx = gaussian_filter(power1, sigma=(sigma_scale, sigma_height))
    Syy = gaussian_filter(power2, sigma=(sigma_scale, sigma_height))
    phase_smooth = gaussian_filter(phase, sigma=(sigma_scale, sigma_height))

    # Compute coherence
    coherence = np.abs(Sxy)**2 / (Sxx * Syy + eps)

    # Vertical smoothing applied separately for each wavelength to reduce 
    # small-scale noise relative to the spectral scale (window ~ λ/2).
    cospectrum_smooth = np.zeros_like(cospectrum)
    cross_power_smooth = np.zeros_like(cross_power)
    for i in range(len(wavelengths)):
        N = max(1, int((0.5 * wavelengths[i]) / dz))
        cospectrum_smooth[i, :] = uniform_filter1d(cospectrum[i, :], size=N)
        cross_power_smooth[i, :] = uniform_filter1d(cross_power[i, :], size=N)
    cospectrum_phys = cospectrum_smooth
    cross_power_phys = cross_power_smooth

    # TESTING # - Estimating empirical phase noise
    noise_threshold = np.percentile(cospectrum_phys, 10)
    noise_mask = cospectrum_phys < noise_threshold
    sigma_phi_empirical = np.nanstd(phase_smooth[noise_mask])
    print(f"Empirical phase noise: {np.degrees(sigma_phi_empirical):.2f} degrees")
    sigma_phi_empirical_profile = np.zeros(len(z_reg))
    for i in range(len(z_reg)):
        noise_col = phase_smooth[:, i][cospectrum_phys[:, i] < 
                    np.percentile(cospectrum_phys[:, i], 10)]
        if len(noise_col) > 5:
            sigma_phi_empirical_profile[i] = np.nanstd(noise_col)
        else:
            sigma_phi_empirical_profile[i] = np.pi / np.sqrt(3)  # pure random phase
    sigma_phi_empirical_profile = sigma_phi_empirical_profile[None, :] 

    # Mask by coherence (currently not used)
    coh_thresh = 0.5
    mask = coherence > coh_thresh
    cospec_masked = np.where(mask, cospectrum_phys, np.nan)

    # Log-scale integration weights
    dlog_lambda = np.gradient(np.log(wavelengths))

    # Broadcast weights
    weights = dlog_lambda[:, None]

    # Momentum flux profile
    F_z = np.nansum(cospectrum_phys * weights, axis=0)
    F_cumulative = np.cumsum(cospectrum_phys * dlog_lambda[:, None], axis=0)
    fraction = cospectrum_phys / F_z[None, :]

    # Horizontal wavelength profile
    # Note: This is still being edited. It is a "point by point" calculation which
    # is very noise-prone. A new method which identifies regions of high cospectral
    # power and averages the phase within these regions is being developed,
    # which should give more robust estimates of the dominant horizontal wavelength.
    lambda_xy = np.full(len(z_reg), np.nan)
    dominant_lambda_z = np.full(len(z_reg), np.nan)
    dominant_phase = np.full(len(z_reg), np.nan)
    peak_power = np.nanmax(cospectrum_phys)

    for i in range(len(z_reg)):
        # Find dominant scale
        dom_idx = np.nanargmax(cospectrum_phys[:, i])

        phi = phase[dom_idx, i]
        phi_uncertainty = sigma_phase[dom_idx, i]

        # Avoid near-zero phase (would give unphysically large λ_xy)
        if np.abs(phi) < 3 * phi_uncertainty:
            continue

        # Use horizontal separation of balloons to get lambda_xy
        dx = dxy_sep_m[i]
        lambda_xy[i] = 2 * np.pi * dx / np.abs(phi) / 1000
        dominant_lambda_z[i] = wavelengths[dom_idx] / 1000
        dominant_phase[i] = phi

    # Compute the maximum theoretical horizontal wavelength that can be resolved given the phase uncertainty
    SNR_threshold = config.get("PHASE_SNR_THRESHOLD", 3)  # Minimum signal-to-noise ratio for reliable phase estimation
    if rs_dict[s1]['file'] != rs_dict[s2]['file']:
        lambda_x_max_km = (2 * np.pi * dxy_sep_m[None, :] / (SNR_threshold * sigma_phase + eps)) / 1000
    else:
        lambda_x_max_km = np.full_like(sigma_phase, np.nan)

    # Estimate lag-1 red noise background for each sonde
    alpha1 = lag1_autocorr(rs_dict[s1]['f_prime_reg'])
    alpha2 = lag1_autocorr(rs_dict[s2]['f_prime_reg'])

    # Background spectrum at each scale (Torrence & Compo eq 16) using the Fourier frequency corresponding to each wavelet scale
    frequencies = 1.0 / wavelengths  # cycles per metre
    P1 = (1 - alpha1**2) / (1 + alpha1**2 - 2*alpha1*np.cos(2*np.pi*frequencies*dz))
    P2 = (1 - alpha2**2) / (1 + alpha2**2 - 2*alpha2*np.cos(2*np.pi*frequencies*dz))

    # Obtain the variance of each signal
    sigma1 = np.nanstd(rs_dict[s1]['f_prime_reg'])
    sigma2 = np.nanstd(rs_dict[s2]['f_prime_reg'])

    # Torrence & Compo eq 31 — 95% significance threshold for cross-wavelet power is
    # Z2_95 = 3.999 for complex wavelet with 2 degrees of freedom
    Z2_95 = 3.999
    threshold = (Z2_95 / 2) * sigma1 * sigma2 * np.sqrt(
        P1[:, None] * P2[:, None])
    # threshold_test = (Z2_95 / 2) * np.sqrt(P1[:, None] * P2[:, None])
    cross_power_unscaled = np.abs(cross_coeffs)
    sig_mask = cross_power_unscaled > threshold

    # Cone of influence (COI) calculation
    dist_from_edge = np.minimum(z_reg - z_reg[0], z_reg[-1] - z_reg)
    coi_scales = dist_from_edge / np.sqrt(2)
    coi_wavelengths = coi_scales * (4 * np.pi) / (m0 + np.sqrt(2 + m0**2))
    coi_wavelengths = np.clip(coi_wavelengths, wavelengths.min(), wavelengths.max())
    coi_kz = (2 * np.pi / coi_wavelengths)

    # Axes
    results_dict["z_reg"] = z_reg
    results_dict["wavelengths"] = wavelengths

    # Results
    results_dict["cospectrum"] = cospectrum_phys
    results_dict["cross_power"] = cross_power_phys
    results_dict["phase"] = phase
    #phase_mask = (np.abs(phase) > np.pi/4) & (np.abs(phase) < 3*np.pi/4)
    #results_dict["phase"] = np.where(phase_mask, phase, np.nan)
    #results_dict["phase"] = np.where(cospectrum_phys > np.percentile(cospectrum_phys, 15), phase, np.nan)
    results_dict["quadrature"] = quadrature
    results_dict["coherence"] = coherence
    results_dict["F_z"] = F_z
    results_dict["F_cumulative"] = F_cumulative
    results_dict["fraction"] = fraction

    # Diagnostics
    results_dict["dxy_sep_km"] = dxy_sep_km
    results_dict["dz_sep"] = dz_sep
    results_dict["t_common"] = t_common
    results_dict["dt_sep"] = dt_sep
    results_dict["lambda_xy"] = lambda_xy
    results_dict["dominant_lambda_z"] = dominant_lambda_z

    # Stokes
    results_dict["I"] = I
    results_dict["D"] = D
    results_dict["P"] = P
    results_dict["Q"] = Q
    results_dict["DoP"] = DoP
    results_dict["theta"] = theta
    results_dict["chi"] = chi

    # Uncertainty
    results_dict["sigma_phase"] = sigma_phase
    results_dict["sigma_phase_deg"] = sigma_phase_deg
    results_dict["lambda_x_max_km"] = lambda_x_max_km
    results_dict["sig_mask"] = sig_mask
    results_dict['coi_kz'] = coi_kz

    return results_dict

def plot_dual_sonde_figs(rs_dict, results, config, p):
    #####################
    ### Plotting Code ###
    #####################
    s1, s2 = list(rs_dict.keys())[:2]

    # plot_field options: "power_rs1", "power_rs2",
    # "cospectrum", "cross_power", "phase", "quadrature"
    plot_field = p
    t_plot = pd.to_datetime(results['t_common'], unit='s')
    alt_mask = (results['z_reg']/1000 >= config['ALT_MIN']) & (results['z_reg']/1000 <= config['ALT_MAX'])

    plot_dict = {
        "power_rs1": {
            "type": "contourf",
            "data": rs_dict[s1]['power_reg'][:, alt_mask],
            "X": 2 * np.pi / results["wavelengths"],
            "Y": results["z_reg"][alt_mask]/1000,
            "cbar": "Wavelet Power (m$^2$ s$^{-2}$)",
            "xlabel": "Vertical Wavenumber $k_z$ (rad/m)",
            "ylabel": "Altitude (km)",
            "tag": "power",
            "title_prefix": "CWT Power Spectrum (Morlet m0=6)",
            "cmap": config["POWERCOLORMAP"],
            "titleserials": rs_dict[s1]['ds'].attrs.get('serial',''),
            "fnameserials": rs_dict[s1]['ds'].attrs.get('serial',''),
            "levels": 50,
            "extend": "max"
        },
        "power_rs2": {
            "type": "contourf",
            "data": rs_dict[s2]['power_reg'][:, alt_mask],
            "X": 2 * np.pi / results["wavelengths"],
            "Y": results["z_reg"][alt_mask]/1000,
            "cbar": "Wavelet Power (m$^2$ s$^{-2}$)",
            "xlabel": "Vertical Wavenumber $k_z$ (rad/m)",
            "ylabel": "Altitude (km)",
            "tag": "power",
            "title_prefix": "CWT Power Spectrum (Morlet m0=6)",
            "cmap": config["POWERCOLORMAP"],
            "titleserials": rs_dict[s2]['ds'].attrs.get('serial',''),
            "fnameserials": rs_dict[s2]['ds'].attrs.get('serial',''),
            "levels": 50,
            "extend": "max"
        },
        "cospectrum": {
            "type": "contourf",
            "data": results["cospectrum"][:, alt_mask],
            "X": 2 * np.pi / results["wavelengths"],
            "Y": results["z_reg"][alt_mask]/1000,
            "cbar": "Wavelet Cospectrum (m$^2$ s$^{-2}$)",
            "xlabel": "Vertical Wavenumber $k_z$ (rad/m)",
            "ylabel": "Altitude (km)",
            "tag": "cospec",
            "title_prefix": "CWT Cospectrum (Morlet m0=6)",
            "cmap": config["POWERCOLORMAP"],
            "titleserials": f"{rs_dict[s1]['ds'].attrs.get('serial','')} AND {rs_dict[s2]['ds'].attrs.get('serial','')}",
            "fnameserials": f"{rs_dict[s1]['ds'].attrs.get('serial','')}_AND_{rs_dict[s2]['ds'].attrs.get('serial','')}",
            "levels": 50,
            "extend": "max"
        },
        "cross_power": {
            "type": "contourf",
            "data": results["cross_power"][:, alt_mask],
            "X": 2 * np.pi / results["wavelengths"],
            "Y": results["z_reg"][alt_mask]/1000,
            "cbar": "Wavelet Cross-Power (m$^2$ s$^{-2}$)",
            "xlabel": "Vertical Wavenumber $k_z$ (rad/m)",
            "ylabel": "Altitude (km)",
            "tag": "xpower",
            "title_prefix": "CWT Cross-Power Spectrum (Morlet m0=6)",
            "cmap": config["POWERCOLORMAP"],
            "titleserials": f"{rs_dict[s1]['ds'].attrs.get('serial','')} AND {rs_dict[s2]['ds'].attrs.get('serial','')}",
            "fnameserials": f"{rs_dict[s1]['ds'].attrs.get('serial','')}_AND_{rs_dict[s2]['ds'].attrs.get('serial','')}",
            "levels": 50,
            "extend": "max"
        },
        "phase": {
            "type": "contourf",
            "data": results["phase"][:, alt_mask],
            "X": 2 * np.pi / results["wavelengths"],
            "Y": results["z_reg"][alt_mask]/1000,
            "cbar": "Phase (rad)",
            "xlabel": "Vertical Wavenumber $k_z$ (rad/m)",
            "ylabel": "Altitude (km)",
            "tag": "phase",
            "title_prefix": "CWT Phase Spectrum (Morlet m0=6)",
            "cmap": config["PHASECOLORMAP"],
            "titleserials": f"{rs_dict[s1]['ds'].attrs.get('serial','')} AND {rs_dict[s2]['ds'].attrs.get('serial','')}",
            "fnameserials": f"{rs_dict[s1]['ds'].attrs.get('serial','')}_AND_{rs_dict[s2]['ds'].attrs.get('serial','')}",
            #"levels": np.arange(-0.3, 0.3, 0.01),
            "levels": np.arange(-np.pi, np.pi, np.pi/31),
            "extend": "both"
        },
        "quadrature": {
            "type": "contourf",
            "data": results["quadrature"][:, alt_mask],
            "X": 2 * np.pi / results["wavelengths"],
            "Y": results["z_reg"][alt_mask]/1000,
            "cbar": "Wavelet Quadrature (m$^2$ s$^{-2}$)",
            "xlabel": "Vertical Wavenumber $k_z$ (rad/m)",
            "ylabel": "Altitude (km)",
            "tag": "quad",
            "title_prefix": "CWT Quadrature Spectrum (Morlet m0=6)",
            "cmap": config["QUADCOLORMAP"],
            "titleserials": f"{rs_dict[s1]['ds'].attrs.get('serial','')} AND {rs_dict[s2]['ds'].attrs.get('serial','')}",
            "fnameserials": f"{rs_dict[s1]['ds'].attrs.get('serial','')}_AND_{rs_dict[s2]['ds'].attrs.get('serial','')}",
            "levels": np.linspace(- np.nanpercentile(np.abs(results["quadrature"]), 99),  np.nanpercentile(np.abs(results["quadrature"]), 99), 51),
            "extend": "both"
        },
        "coherence": {
            "type": "contourf",
            "data": results["coherence"][:, alt_mask],
            "X": 2 * np.pi / results["wavelengths"],
            "Y": results["z_reg"][alt_mask]/1000,
            "cbar": "Wavelet Coherence",
            "xlabel": "Vertical Wavenumber $k_z$ (rad/m)",
            "ylabel": "Altitude (km)",
            "tag": "coherence",
            "title_prefix": "CWT Coherence (Morlet m0=6)",
            "cmap": config["POWERCOLORMAP"],
            "titleserials": f"{rs_dict[s1]['ds'].attrs.get('serial','')} AND {rs_dict[s2]['ds'].attrs.get('serial','')}",
            "fnameserials": f"{rs_dict[s1]['ds'].attrs.get('serial','')}_AND_{rs_dict[s2]['ds'].attrs.get('serial','')}",
            "levels": np.linspace(0, 1, 51),
            "extend": "neither"
        },
        "fraction": {
            "type": "contourf",
            "data": results["fraction"][:, alt_mask],
            "X": 2 * np.pi / results["wavelengths"],
            "Y": results["z_reg"][alt_mask]/1000,
            "cbar": "Momentum Flux (m$^2$ s$^{-2}$)",
            "xlabel": "Vertical Wavenumber $k_z$ (rad/m)",
            "ylabel": "Altitude (km)",
            "tag": "Fz",
            "title_prefix": "Estimated Momentum Flux Profile",
            "cmap": config["POWERCOLORMAP"],
            "titleserials": f"{rs_dict[s1]['ds'].attrs.get('serial','')} AND {rs_dict[s2]['ds'].attrs.get('serial','')}",
            "fnameserials": f"{rs_dict[s1]['ds'].attrs.get('serial','')}_AND_{rs_dict[s2]['ds'].attrs.get('serial','')}",
            "levels": 50,
            "extend": "max"
        },
        "Fz": {
            "type": "line_profile",
            "data": results["F_z"][alt_mask],
            "X": None,
            "Y": results["z_reg"][alt_mask]/1000,
            "cbar": None,
            "xlabel": "Momentum Flux (m$^2$ s$^{-2}$)",
            "ylabel": "Altitude (km)",
            "tag": "Fz_profile",
            "title_prefix": "Estimated Momentum Flux Profile",
            "cmap": None,
            "titleserials": f"{rs_dict[s1]['ds'].attrs.get('serial','')} AND {rs_dict[s2]['ds'].attrs.get('serial','')}",
            "fnameserials": f"{rs_dict[s1]['ds'].attrs.get('serial','')}_AND_{rs_dict[s2]['ds'].attrs.get('serial','')}",
            "levels": None,
            "extend": None
        },
        "dxy_sep": {
            "type": "line_profile",
            "data": results["dxy_sep_km"][alt_mask],
            "cbar": None,
            "X": None,
            "Y": results["z_reg"][alt_mask]/1000,
            "xlabel": "Horizontal Separation (km)",
            "ylabel": "Altitude (km)",
            "tag": "dxy_sep",
            "title_prefix": "Horizontal Separation Profile",
            "cmap": None,
            "titleserials": f"{rs_dict[s1]['ds'].attrs.get('serial','')} AND {rs_dict[s2]['ds'].attrs.get('serial','')}",
            "fnameserials": f"{rs_dict[s1]['ds'].attrs.get('serial','')}_AND_{rs_dict[s2]['ds'].attrs.get('serial','')}",
            "levels": None,
            "extend": None
        },
        "dz_sep": {
            "type": "line_time_series",
            "data": results["dz_sep"],
            "X": t_plot,
            "Y": None,
            "cbar": None,
            "xlabel": "Time",
            "ylabel": "Vertical Separation (m)",
            "tag": "dz_sep",
            "title_prefix": "Vertical Separation Profile",
            "cmap": None,
            "titleserials": f"{rs_dict[s1]['ds'].attrs.get('serial','')} AND {rs_dict[s2]['ds'].attrs.get('serial','')}",
            "fnameserials": f"{rs_dict[s1]['ds'].attrs.get('serial','')}_AND_{rs_dict[s2]['ds'].attrs.get('serial','')}",
            "levels": None,
            "extend": None
        },
        "dt_sep": {
            "type": "line_profile",
            "data": results["dt_sep"][alt_mask]/60,
            "cbar": None,
            "X": None,
            "Y": results["z_reg"][alt_mask]/1000,
            "xlabel": "'Time lag Δt (minutes)'",
            "ylabel": "Altitude (km)",
            "tag": "dt_sep",
            "title_prefix": "Temporal Separation Profile",
            "cmap": None,
            "titleserials": f"{rs_dict[s1]['ds'].attrs.get('serial','')} AND {rs_dict[s2]['ds'].attrs.get('serial','')}",
            "fnameserials": f"{rs_dict[s1]['ds'].attrs.get('serial','')}_AND_{rs_dict[s2]['ds'].attrs.get('serial','')}",
            "levels": None,
            "extend": None
        },
        "lambda_xy": {
            "type": "line_profile",
            "data": results["lambda_xy"][alt_mask],
            "cbar": None,
            "X": None,
            "Y": results["z_reg"][alt_mask]/1000,
            "xlabel": "Dominant Horizontal Wavelength (km)",
            "ylabel": "Altitude (km)",
            "tag": "lambda_xy",
            "title_prefix": "Horizontal Wavelength Profile",
            "cmap": None,
            "titleserials": f"{rs_dict[s1]['ds'].attrs.get('serial','')} AND {rs_dict[s2]['ds'].attrs.get('serial','')}",
            "fnameserials": f"{rs_dict[s1]['ds'].attrs.get('serial','')}_AND_{rs_dict[s2]['ds'].attrs.get('serial','')}",
            "levels": None,
            "extend": None
        },
        "I": {
            "type": "contourf",
            "data": results["I"][:, alt_mask],
            "X": 2 * np.pi / results["wavelengths"],
            "Y": results["z_reg"][alt_mask]/1000,
            "cbar": "Stokes I (m$^2$ s$^{-2}$)",
            "xlabel": "Vertical Wavenumber $k_z$ (rad/m)",
            "ylabel": "Altitude (km)",
            "tag": "Stokes_I",
            "title_prefix": "Stokes Parameter I - Total Intensity",
            "cmap": config["POWERCOLORMAP"],
            "titleserials": f"{rs_dict[s1]['ds'].attrs.get('serial','')} AND {rs_dict[s2]['ds'].attrs.get('serial','')}",
            "fnameserials": f"{rs_dict[s1]['ds'].attrs.get('serial','')}_AND_{rs_dict[s2]['ds'].attrs.get('serial','')}",
            "levels": 50,
            "extend": "max"
        },
        "D": {
            "type": "contourf",
            "data": results["D"][:, alt_mask],
            "X": 2 * np.pi / results["wavelengths"],
            "Y": results["z_reg"][alt_mask]/1000,
            "cbar": "Stokes D (m$^2$ s$^{-2}$)",
            "xlabel": "Vertical Wavenumber $k_z$ (rad/m)",
            "ylabel": "Altitude (km)",
            "tag": "Stokes_D",
            "title_prefix": "Stokes Parameter D - Polarisation Asymmetry",
            "cmap": config["QUADCOLORMAP"],
            "titleserials": f"{rs_dict[s1]['ds'].attrs.get('serial','')} AND {rs_dict[s2]['ds'].attrs.get('serial','')}",
            "fnameserials": f"{rs_dict[s1]['ds'].attrs.get('serial','')}_AND_{rs_dict[s2]['ds'].attrs.get('serial','')}",
            "levels": np.linspace(- np.nanpercentile(np.abs(results["quadrature"]), 99.9),  np.nanpercentile(np.abs(results["quadrature"]), 99.9), 51),
            "extend": "both"
        },
        "P": {
            "type": "contourf",
            "data": results["P"][:, alt_mask],
            "X": 2 * np.pi / results["wavelengths"],
            "Y": results["z_reg"][alt_mask]/1000,
            "cbar": "Stokes P (m$^2$ s$^{-2}$)",
            "xlabel": "Vertical Wavenumber $k_z$ (rad/m)",
            "ylabel": "Altitude (km)",
            "tag": "Stokes_P",
            "title_prefix": "Stokes Parameter P - In-phase Correlation",
            "cmap": config["POWERCOLORMAP"],
            "titleserials": f"{rs_dict[s1]['ds'].attrs.get('serial','')} AND {rs_dict[s2]['ds'].attrs.get('serial','')}",
            "fnameserials": f"{rs_dict[s1]['ds'].attrs.get('serial','')}_AND_{rs_dict[s2]['ds'].attrs.get('serial','')}",
            "levels": 50,
            "extend": "both"
        },
        "Q": {
            "type": "contourf",
            "data": results["Q"][:, alt_mask],
            "X": 2 * np.pi / results["wavelengths"],
            "Y": results["z_reg"][alt_mask]/1000,
            "cbar": "Stokes Q (m$^2$ s$^{-2}$)",
            "xlabel": "Vertical Wavenumber $k_z$ (rad/m)",
            "ylabel": "Altitude (km)",
            "tag": "Stokes_Q",
            "title_prefix": "Stokes Parameter Q - Out-of-phase Correlation",
            "cmap": config["QUADCOLORMAP"],
            "titleserials": f"{rs_dict[s1]['ds'].attrs.get('serial','')} AND {rs_dict[s2]['ds'].attrs.get('serial','')}",
            "fnameserials": f"{rs_dict[s1]['ds'].attrs.get('serial','')}_AND_{rs_dict[s2]['ds'].attrs.get('serial','')}",
            "levels": np.linspace(- np.nanpercentile(np.abs(results["quadrature"]), 99.9),  np.nanpercentile(np.abs(results["quadrature"]), 99.9), 51),
            "extend": "both"
        },
        "DoP": {
            "type": "contourf",
            "data": results["DoP"][:, alt_mask],
            "X": 2 * np.pi / results["wavelengths"],
            "Y": results["z_reg"][alt_mask]/1000,
            "cbar": "Degree of Polarisation",
            "xlabel": "Vertical Wavenumber $k_z$ (rad/m)",
            "ylabel": "Altitude (km)",
            "tag": "Stokes_DoP",
            "title_prefix": "Degree of Polarisation",
            "cmap": config["POWERCOLORMAP"],
            "titleserials": f"{rs_dict[s1]['ds'].attrs.get('serial','')} AND {rs_dict[s2]['ds'].attrs.get('serial','')}",
            "fnameserials": f"{rs_dict[s1]['ds'].attrs.get('serial','')}_AND_{rs_dict[s2]['ds'].attrs.get('serial','')}",
            "levels": 51,
            "extend": "neither"
        },
        "theta": {
            "type": "contourf",
            "data": results["theta"][:, alt_mask],
            "X": 2 * np.pi / results["wavelengths"],
            "Y": results["z_reg"][alt_mask]/1000,
            "cbar": "Orientation Angle θ (rad)",
            "xlabel": "Vertical Wavenumber $k_z$ (rad/m)",
            "ylabel": "Altitude (km)",
            "tag": "Stokes_theta",
            "title_prefix": "Orientation Angle θ",
            "cmap": config["PHASECOLORMAP"],
            "titleserials": f"{rs_dict[s1]['ds'].attrs.get('serial','')} AND {rs_dict[s2]['ds'].attrs.get('serial','')}",
            "fnameserials": f"{rs_dict[s1]['ds'].attrs.get('serial','')}_AND_{rs_dict[s2]['ds'].attrs.get('serial','')}",
            "levels": np.arange(-np.pi/2, np.pi/2, np.pi/20),
            "extend": "both"
        },
        "chi": {
            "type": "contourf",
            "data": results["chi"][:, alt_mask],
            "X": 2 * np.pi / results["wavelengths"],
            "Y": results["z_reg"][alt_mask]/1000,
            "cbar": "Ellipticity Angle χ (rad)",
            "xlabel": "Vertical Wavenumber $k_z$ (rad/m)",
            "ylabel": "Altitude (km)",
            "tag": "Stokes_chi",
            "title_prefix": "Ellipticity Angle χ",
            "cmap": config["PHASECOLORMAP"],
            "titleserials": f"{rs_dict[s1]['ds'].attrs.get('serial','')} AND {rs_dict[s2]['ds'].attrs.get('serial','')}",
            "fnameserials": f"{rs_dict[s1]['ds'].attrs.get('serial','')}_AND_{rs_dict[s2]['ds'].attrs.get('serial','')}",
            "levels": np.arange(-np.pi/4, np.pi/4+np.pi/20, np.pi/20),
            "extend": "both"
        },
        "sigma_phase": {
            "type": "contourf",
            "data": results["sigma_phase"][:, alt_mask],
            "X": 2 * np.pi / results["wavelengths"],
            "Y": results["z_reg"][alt_mask]/1000,
            "cbar": "Phase Uncertainty (rad)",
            "xlabel": "Vertical Wavenumber $k_z$ (rad/m)",
            "ylabel": "Altitude (km)",
            "tag": "Stokes_sigma_phase",
            "title_prefix": "Phase Uncertainty",
            "cmap": config["PHASECOLORMAP"],
            "titleserials": f"{rs_dict[s1]['ds'].attrs.get('serial','')} AND {rs_dict[s2]['ds'].attrs.get('serial','')}",
            "fnameserials": f"{rs_dict[s1]['ds'].attrs.get('serial','')}_AND_{rs_dict[s2]['ds'].attrs.get('serial','')}",
            "levels": np.linspace(0, np.nanpercentile(results["sigma_phase"], 99.9), 51),
            "extend": "max"
        },
        "sigma_phase_deg": {
            "type": "contourf",
            "data": results["sigma_phase_deg"][:, alt_mask],
            "X": 2 * np.pi / results["wavelengths"],
            "Y": results["z_reg"][alt_mask]/1000,
            "cbar": "Phase Uncertainty (degrees)",
            "xlabel": "Vertical Wavenumber $k_z$ (rad/m)",
            "ylabel": "Altitude (km)",
            "tag": "Stokes_sigma_phase_deg",
            "title_prefix": "Phase Uncertainty",
            "cmap": config["PHASECOLORMAP"],
            "titleserials": f"{rs_dict[s1]['ds'].attrs.get('serial','')} AND {rs_dict[s2]['ds'].attrs.get('serial','')}",
            "fnameserials": f"{rs_dict[s1]['ds'].attrs.get('serial','')}_AND_{rs_dict[s2]['ds'].attrs.get('serial','')}",
            "levels": np.linspace(0, np.nanpercentile(results["sigma_phase_deg"], 99.9), 51),
            "extend": "max"
        },
        "lambda_x_max_km": {
            "type": "contourf",
            "data": results["lambda_x_max_km"][:, alt_mask],
            "X": 2 * np.pi / results["wavelengths"],
            "Y": results["z_reg"][alt_mask]/1000,
            "cbar": "Maximum Resolvable Horizontal Wavelength (km)",
            "xlabel": "Vertical Wavenumber $k_z$ (rad/m)",
            "ylabel": "Altitude (km)",
            "tag": "lambda_x_max_km",
            "title_prefix": "Maximum Resolvable Horizontal Wavelength",
            "cmap": config["POWERCOLORMAP"],
            "titleserials": f"{rs_dict[s1]['ds'].attrs.get('serial','')} AND {rs_dict[s2]['ds'].attrs.get('serial','')}",
            "fnameserials": f"{rs_dict[s1]['ds'].attrs.get('serial','')}_AND_{rs_dict[s2]['ds'].attrs.get('serial','')}",
            "levels": 50,
            "extend": "max"
        }
    }

    plot_cfg = plot_dict[plot_field]

    plot_data = plot_cfg["data"]
    cbar_label = plot_cfg["cbar"]
    tag = plot_cfg["tag"]
    title_prefix = plot_cfg["title_prefix"]
    cmap = plot_cfg["cmap"]
    titleserials = plot_cfg["titleserials"]
    fnameserials = plot_cfg["fnameserials"]
    levels = plot_cfg["levels"]
    extend = plot_cfg["extend"]

    fig, ax1 = plt.subplots(figsize=(8,8))
    plt.subplots_adjust(left=0.1, right=0.9)
    # Bottom x-axis: k_z

    if plot_cfg["type"] == "line_profile":
        ax1.plot(plot_data, plot_cfg["Y"], color='red', linewidth=2)
        ax1.set_xlabel(plot_cfg["xlabel"])
        ax1.set_ylabel(plot_cfg["ylabel"])
        plt.title(
            f"{title_prefix}\n"
            f"{rs_dict[s1]['ds'].attrs['site_name']}: {titleserials} ({rs_dict[s1]['ds'].attrs.get('source','')})\n"
            f"{pd.to_datetime(rs_dict[s1]['ds'].attrs['launch_time']):%d %b %Y %H:%M:%S UTC}",
            fontsize = 10
        )
        plt.grid(True, linestyle='--', alpha=0.5)
        os.makedirs("plots", exist_ok=True)
        plt.savefig(os.path.join("plots", f"{pd.to_datetime(rs_dict[s1]['ds'].attrs['launch_time']).strftime('%Y%m%d_%H%M%S')}_{fnameserials}_{tag}.png"), dpi=225, bbox_inches='tight', pad_inches=0.02)
        print(f"Saved figure {fnameserials}_{tag}.png in directory {os.path.join(os.getcwd(), 'plots')} using plot_dual_sonde_figs")
        plt.close()
        return
    
    elif plot_cfg["type"] == "line_time_series":
        t_plot = pd.to_datetime(results['t_common'], unit='s')
        ax1.plot(plot_cfg["X"], plot_data, color='red', linewidth=2)
        ax1.set_xlabel('Time')
        ax1.set_ylabel(plot_cfg["ylabel"])
        ax1.set_title(
            f"{title_prefix}\n"
            f"{rs_dict[s1]['ds'].attrs['site_name']}: {titleserials} ({rs_dict[s1]['ds'].attrs.get('source','')})\n"
            f"{pd.to_datetime(rs_dict[s1]['ds'].attrs['launch_time']):%d %b %Y %H:%M:%S UTC}",
            fontsize = 10
        )
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        plt.grid(True, linestyle='--', alpha=0.5)
        os.makedirs("plots", exist_ok=True)
        plt.savefig(os.path.join("plots", f"{pd.to_datetime(rs_dict[s1]['ds'].attrs['launch_time']).strftime('%Y%m%d_%H%M%S')}_{fnameserials}_{tag}.png"), dpi=225, bbox_inches='tight', pad_inches=0.02)
        print(f"Saved figure {fnameserials}_{tag}.png in directory {os.path.join(os.getcwd(), 'plots')} using plot_dual_sonde_figs")
        plt.close()
        return
    
    elif plot_cfg["type"] == "contourf":
        k_z = 2 * np.pi / results['wavelengths']  # rad/m
        cf = ax1.contourf(
            k_z,       # x-axis: vertical wavenumber
            plot_cfg["Y"],       # y-axis: altitude
            plot_data.T,   # transpose so shape matches (alt, k_z)
            levels=levels,
            cmap=cmap,
            extend=extend
        )

        ## Adding relevant masks and hatching for significance and phase uncertainty
        if plot_field in ["cospectrum", "cross_power"]:
            # Create 2D grids matching the shape of sig_mask
            kz_2d, z_2d = np.meshgrid(k_z, plot_cfg["Y"], indexing='ij')
            
            # Apply alt_mask to sig_mask along altitude axis
            sig_mask_masked = results['sig_mask'][:, alt_mask]

            interp = RegularGridInterpolator(
                (k_z, plot_cfg["Y"]),
                (~sig_mask_masked).astype(float),
                bounds_error=False,
                fill_value=0
            )

            kz_fine = np.linspace(kz_2d.min(), kz_2d.max(), 5000)
            z_fine = np.linspace(z_2d.min(), z_2d.max(), 5000)
            KZ_fine, Z_fine = np.meshgrid(kz_fine, z_fine, indexing='ij')
            mask_fine = interp((KZ_fine, Z_fine))
            mask_smooth = gaussian_filter(mask_fine, sigma=30.0)
            
            # Stipple where NOT significant
            not_sig = ~sig_mask_masked
            plt.rcParams['hatch.linewidth'] = 0.3
            """ax1.contourf(
                kz_2d,
                z_2d,
                (~sig_mask_masked).astype(int),
                levels=[0.5, 1.5],
                colors='none',
                hatches=['xxxxxx'],
                linewidths=0,
            )"""
            ax1.contourf(
                KZ_fine,
                Z_fine,
                mask_smooth,
                levels=[0.5, 1.5],
                colors='none',
                hatches=['xxxxxx'],
                linewidths=0,
            )
            for collection in ax1.collections:
                collection.set_edgecolor('white')
            for collection in ax1.collections:
                collection.set_linewidth(0.)

        if plot_field == "phase":
            #mask_data = (results['cospectrum'] <= np.percentile(results['cospectrum'], 15)).astype(float)
            mask_data = (results['cross_power'] <= np.percentile(results['cross_power'], 15)).astype(float)
            mask_smooth = uniform_filter(mask_data, size=3)
            ax1.contourf(
                k_z,
                results['z_reg'] / 1000.0,
                mask_smooth.T,
                levels=[0.3, 1.5],
                colors=['k'],
                alpha=0.7
            )

        # COI overlay
        coi_kz_masked = results['coi_kz'][alt_mask]
        z_km_masked = results['z_reg'][alt_mask] / 1000

        ax1.fill_betweenx(
            z_km_masked,
            k_z.min(),
            np.clip(coi_kz_masked, k_z.min(), k_z.max()),
            color='gray',
            alpha=0.3,
            zorder=6
        )
        ax1.plot(
            np.clip(coi_kz_masked, k_z.min(), k_z.max()),
            z_km_masked,
            color='white',
            linewidth=1,
            linestyle='--',
            zorder=7
        )

        ax1.set_xlabel('Vertical Wavenumber $k_z$ (rad/m)')
        ax1.set_ylabel('Altitude (km)')
        ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))

        # --- Bottom axis (wavenumber) ---
        ax1.set_xscale('log')
        ax1.set_xlim(k_z.min(), k_z.max())

        # Clean ticks
        lambda_ticks_km = np.array([15, 10, 5, 3, 2, 1.5, 1])
        lambda_ticks_m = lambda_ticks_km * 1000
        #k_ticks = 2 * np.pi / lambda_ticks_m
        k_ticks = np.array([0.0005, 0.001, 0.002, 0.005])
        ax1.set_xticks(k_ticks)
        ax1.set_xticklabels([
            r"$%.1f\times10^{%d}$" % (kt/10**np.floor(np.log10(kt)), np.floor(np.log10(kt)))
            for kt in k_ticks
        ])
        #ax1.set_xticklabels([f'{kt:.1e}' for kt in k_ticks])
        ax1.set_xlabel('Vertical Wavenumber $k_z$ (rad/m)')

        # Top axis (wavelength)
        ax2 = ax1.twiny()
        ax2.set_xscale('log')
        ax2.set_xlim(ax1.get_xlim())

        k_top_ticks = 2 * np.pi / lambda_ticks_m
        ax2.set_xticks(k_top_ticks)
        ax2.set_xticklabels([f'{lt:g}' for lt in lambda_ticks_km])
        ax2.set_xlabel('Vertical Wavelength $\\lambda_z$ (km)')

        # Remove messy minor ticks
        ax1.minorticks_off()
        ax2.minorticks_off()

        cbar = plt.colorbar(cf, ax=ax1, label=cbar_label)
        #if plot_field not in ["phase", "quadrature", "coherence", "Fz"]:
        #    cbar.ax.yaxis.set_major_formatter(mticker.FuncFormatter(four_digit_formatter))

        # Extract launch time
        t = pd.to_datetime(rs_dict[s1]['ds'].attrs["launch_time"])
        timestamp = t.strftime("%Y%m%d_%H%M%S")
        prettytimestamp = t.strftime("%d %b %Y %H:%M:%S UTC")

        # Build figure name
        figname = (
            f"{timestamp}_"
            f"{fnameserials}_"
            f"{tag}.png"
        )   
        plt.title(
            f"{title_prefix}\n"
            f"{rs_dict[s1]['ds'].attrs['site_name']}: {titleserials} ({rs_dict[s1]['ds'].attrs.get('source','')})\n"
            f"{t:%d %b %Y %H:%M:%S UTC}",
            fontsize = 10
        )

        # Ensure the 'plots' subdirectory exists (create if needed)
        os.makedirs("plots", exist_ok=True)
        plot_path = os.path.join("plots", figname)
        plt.savefig(plot_path, dpi=225, bbox_inches='tight', pad_inches=0.02)
        print(f"Saved figure {figname} in directory {os.path.join(os.getcwd(), 'plots')} using plot_dual_sonde_figs")

        plt.close()
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 8), sharey=True)
    plt.subplots_adjust(wspace=0.1)

    # Panel 1: horizontal wavelength
    axes[0].plot(results['lambda_xy'], results['z_reg']/1000, color='steelblue', linewidth=2)
    axes[0].set_xlabel('Estimated $\\lambda_x$ (km)')
    axes[0].set_ylabel('Altitude (km)')
    axes[0].set_title('Horizontal Wavelength')
    axes[0].grid(True, linestyle='--', alpha=0.5)
    axes[0].set_xlim(left=0)

    # Panel 2: dominant vertical wavelength
    axes[1].plot(results['dominant_lambda_z'], results['z_reg']/1000, color='darkorange', linewidth=2)
    axes[1].set_xlabel('Dominant $\\lambda_z$ (km)')
    axes[1].set_title('Dominant Vertical Wavelength')
    axes[1].grid(True, linestyle='--', alpha=0.5)

    # Panel 3: sonde separation for context
    axes[2].plot(results['dxy_sep_km'], results['z_reg']/1000, color='gray', linewidth=1.5, linestyle='--')
    axes[2].set_xlabel('Sonde separation (km)')
    axes[2].set_title('Horizontal Separation')
    axes[2].grid(True, linestyle='--', alpha=0.5)

    t = pd.to_datetime(rs_dict[s1]['ds'].attrs['launch_time'])
    s1_serial = rs_dict[s1]['ds'].attrs.get('serial', '')
    s2 = list(rs_dict.keys())[1]
    s2_serial = rs_dict[s2]['ds'].attrs.get('serial', '')

    plt.suptitle(
        f'Estimated Horizontal GW Wavelength Profile\n'
        f'{rs_dict[s1]["ds"].attrs["site_name"]}: {s1_serial} AND {s2_serial}\n'
        f'{t:%d %b %Y %H:%M:%S UTC}',
        fontsize=10
    )

    os.makedirs('plots', exist_ok=True)
    timestamp = t.strftime('%Y%m%d_%H%M%S')
    figname = f'{timestamp}_{s1_serial}_AND_{s2_serial}_horizontal_wavelength.png'
    plt.savefig(os.path.join('plots', figname), dpi=225, bbox_inches='tight', pad_inches=0.02)
    print(f'Saved {figname}')
    plt.close()

def main(args):
    # Resolve sonde filter variables
    filter_variables = args.filter_variable
    if len(filter_variables) == 1:
        # Same filter variable for all sondes
        filter_variables = filter_variables * len(args.files)
    elif len(filter_variables) != len(args.files):
        raise ValueError(f"Number of filter variables ({len(filter_variables)}) must be 1 or match number of files ({len(args.files)})")

    # Plotting configuration
    config = {
        "BUTTERWORTH_CUTOFF_WAVELENGTH": 15,  # high-pass filter cutoff (km)
        "BUTTER_ORDER": 4,
        "ESTIMATED_GW_WAVELENGTH_FOR_RUNNING_RMS": 5, # becomes window length, L, for boxcar (km)
        "POLY_ORDER": 4,
        "MORLET_CENTRAL_WAVENUMBER_M_0": 6,
        "POWERCOLORMAP": cm.get_cmap('viridis'),
        "PHASECOLORMAP": cm.get_cmap('twilight_shifted'),
        "QUADCOLORMAP": cm.get_cmap('seismic'),
        "ALT_MIN": 0.0,
        "ALT_MAX": 40.0,
        "PHASE_SNR_THRESHOLD": 3.0
    }

    # Keep track of created figures
    created_figures = []

    # Deal with 'all' option for plot_variable
    if 'all' in args.plot_variable:
        args.plot_variable = [
            'power_rs1', 'power_rs2',
            'cospectrum', 'cross_power',
            'phase', 'quadrature',
            'coherence', 'Fz', 'fraction',
            'dxy_sep', 'dz_sep', 'dt_sep',
            'lambda_xy', 'I', 'D', 'P', 'Q',
            'DoP', 'theta', 'chi', 'sigma_phase', 'sigma_phase_deg',
            'lambda_x_max_km'
        ]

    # Process given launches
    if args.files:
        rs_dict = {}
        for f, fvar in zip(args.files, filter_variables):
            if os.path.isdir(f):
                 # If a single directory is given
                for root, _, files in os.walk(f):
                    for file in files:
                        filepath = os.path.join(root, file)
                        # Processing radiosonde data from file
                        print(f"Processing file: {filepath}")
                        rs_dict = initialise_rs_dict(filepath, rs_dict)
                        rs_dict = run_butterworth_filter(rs_dict, config, fvar)
                        rs_dict = compute_single_spectrum(filepath, rs_dict, config, args)
                        created_figures.append(filepath)
            else:
                # Processing single radiosonde data from file
                print(f"Processing file: {f}")
                rs_dict = initialise_rs_dict(f, rs_dict)
                rs_dict = run_butterworth_filter(rs_dict, config, fvar)
                rs_dict = compute_single_spectrum(f, rs_dict, config, args)
                created_figures.append(f)

        if len(args.files) == 2 and not os.path.isdir(args.files[0]) and not os.path.isdir(args.files[1]):
            f1, f2 = args.files[0], args.files[1]
            ds1 = xr.open_dataset(f1)
            ds2 = xr.open_dataset(f2)
            t1 = pd.to_datetime(ds1.attrs["launch_time"])
            t2 = pd.to_datetime(ds2.attrs["launch_time"])
            within_90_min = abs(t1 - t2) <= pd.Timedelta(minutes=90)
            if within_90_min == True:
                print(f"Processing files: {f1} and {f2}, which are within 90 minutes of each other, to create co-spectra plot(s).")
                results = compute_cospectra(rs_dict, config)
                for p in args.plot_variable:
                    plot_dual_sonde_figs(rs_dict, results, config, p)

            ds1.close()
            ds2.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Radiosonde CWT Spectrum Creator"
    )

    parser.add_argument(
        '-f', '--files',
        nargs='+',
        required=True,
        help='One or more local file paths or directories'
    )

    parser.add_argument(
        '-p', '--plot-variable',
        nargs='+',
        choices=['power_rs1', 'power_rs2', 'cospectrum', 'cross_power', 'phase', 'quadrature', 'coherence', 'Fz', 'fraction', 'dxy_sep', 'dz_sep', 'dt_sep', 'lambda_xy', 'I', 'D', 'P', 'Q', 'DoP', 'theta', 'chi', 'sigma_phase', 'sigma_phase_deg', 'lambda_x_max_km', 'all'],
        default=['cospectrum'],
        help='Variable(s) to plot (default: cospectrum)'
    )

    parser.add_argument(
        '-v', '--filter-variable',
        nargs='+',
        choices=['u', 'v', 'w', 'vel_h'],  # whatever variables you support
        default=['u'],
        help='Variable(s) to filter for each sonde (default: u for all). '
            'Specify one value to use for all sondes, or one per file.'
    )

    args = parser.parse_args()

    main(args)
