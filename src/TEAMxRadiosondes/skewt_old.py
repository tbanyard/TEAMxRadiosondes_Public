import os
import xarray as xr
import netCDF4 as nc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import datetime
from pathlib import Path
import sys
import scipy.stats as stats
import time
import argparse
from TEAMxRadiosondes.funcs import dry_adiabat, moist_adiabat, moist_adiabat_metpy
from metpy.plots import SkewT
from metpy.units import units

def createskewtmetpy(f, args):
    ds = xr.load_dataset(f)

    # 1. Extract and assign MetPy units (Crucial for correct adiabats)
    p = ds["pressure"].values * units.hPa
    t = (ds["temp"].values - 273.15) * units.degC
    td = (ds["dewp"].values - 273.15) * units.degC

    # Handle pressure units if they were in Pa
    if np.nanmax(p) > 2000 * units.hPa:
        p = p / 100.0

    # 2. Setup the Figure and SkewT object
    fig = plt.figure(figsize=(9, 9))
    skew = SkewT(fig, rotation=45) # Standard 45-degree skew

    # 3. Plot the data
    skew.plot(p, t, 'r', linewidth=2, label='Temperature')
    skew.plot(p, td, 'b', linewidth=2, label='Dewpoint')

    # 4. Add standard background lines
    skew.plot_dry_adiabats(t0=np.arange(-60, 150, 10) * units.degC, 
                           color='brown', linewidth=0.5, alpha=0.5)
    
    # This fixes your moist adiabat "left-drift" issue automatically
    skew.plot_moist_adiabats(t0=np.arange(-40, 50, 5) * units.degC, 
                             color='green', linewidth=0.5, alpha=0.5)
    
    skew.plot_mixing_lines(color='black', linestyle=':', alpha=0.3)

    # 5. Set axis limits and labels
    skew.ax.set_ylim(1000, 50)  # Pressure range
    skew.ax.set_xlim(-60, 40)   # Temperature range at 1000hPa
    
    plt.title(f"{ds.attrs.get('serial','')} {ds.attrs.get('startdate','')}", loc='left')
    plt.legend(loc='upper left', frameon=True)
    
    # Add your specific text box
    skew.ax.text(0.02, 0.02, "Note: Quicklook Use Only.\nDewpoint data may be unreliable.",
                 transform=skew.ax.transAxes, fontsize=8, bbox=dict(facecolor='white', alpha=1.0))

    # Save logic (keep your existing pathing)
    plt.savefig(f"plots/skewt_metpy_{ds.attrs.get('serial','')}.png", dpi=300)
    plt.close()

def createskewt(f, args):
    ds = xr.load_dataset(f)

    # Extract arrays
    P = ds["pressure"].values
    T = ds["temp"].values - 273.15 # In Celsius
    Td = ds["dewp"].values - 273.15 # In Celsius

    # --- Ensure pressure is in hPa ---
    if np.nanmax(P) > 2000:
        P = P / 100.0

    # --- Choose log base and skew ---
    use_log10 = True      # set False for natural log scale
    if use_log10:
        Y = np.log10(P / 1000.0)         # 0 at 1000 hPa
        skew = 78.0                      # good tilt
    else:
        Y = np.log(P / 1000.0)
        skew = 34.744                      # ≈ 80/ln(10)

    # --- Transform x for skew-T plotting ---
    XT = T - (skew * Y)
    XD = Td - (skew * Y)

    # ---------------------------------------------------------------------------
    # Figure setup
    # ---------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_title(f"{ds.attrs.get('serial','')}  {ds.attrs.get('startdate','')} {ds.attrs.get('starttime','')}  ({ds.attrs.get('source','')})".strip(), fontsize=10)

    # --- Plot temperature and dew-point profiles ---
    ax.plot(XT, Y, color='r', linewidth=1, label="Temp")
    ax.plot(XD, Y, color='b', linewidth=1, label="Dewpoint")

    # ---------------------------------------------------------------------------
    # Diagonal isotherms (every 10 °C)
    # ---------------------------------------------------------------------------
    Tiso = np.arange(-200, 55, 10) # Winter + Summer EOP
    Pgrid = np.linspace(1000, 20, 60)
    Yg = np.log10(Pgrid / 1000.0) if use_log10 else np.log(Pgrid / 1000.0)

    for t in Tiso:
        Xiso = t - (skew * Yg)
        ax.plot(Xiso, Yg, color='0.5', linestyle='-', linewidth=0.25)

    # ---------------------------------------------------------------------------
    # Pressure (isobar) lines and ticks
    # ---------------------------------------------------------------------------
    pticks = np.array([50, 100, 150, 200, 250, 300, 400, 500,
                    600, 700, 800, 900, 1000])
    yticks = np.log10(pticks / 1000.0) if use_log10 else np.log(pticks / 1000.0)
    ax.set_yticks(yticks)
    ax.set_yticklabels([str(int(p)) for p in pticks])
    ax.invert_yaxis()

    # Horizontal pressure lines
    for p in pticks:
        Yp = np.log10(p / 1000.0) if use_log10 else np.log(p / 1000.0)
        ax.plot(ax.get_xlim(), [Yp, Yp],
                color='0.5', linestyle='-', linewidth=0.25)

    # ---------------------------------------------------------------------------
    # Dry & moist adiabats
    # ---------------------------------------------------------------------------
    """def dry_adiabat(T0, Pstart=1000.0):
        # Return temperature along a dry adiabat starting at T0 (°C) and Pstart (hPa).
        Rd_cp = 0.286  # R_d / c_p
        P = np.linspace(Pstart, 20, 100)
        T = (T0 + 273.15) * (P / Pstart) ** Rd_cp - 273.15
        return T, P"""

    """def moist_adiabat(T0, Pstart=1000.0):
        # Crude moist-adiabat for plotting; not thermodynamically exact.
        P = np.linspace(Pstart, 20, 100)
        #T = T0 - 6 * np.log(Pstart / P)  # very rough slope
        T = (T0 + 273.15) * (P / Pstart) ** (287.0 / 1005.0) - (2.5e6 * 0.622 * 6.112 *
        np.exp((17.67 * T0) / (T0 + 243.5)) / (1005.0 * (Pstart * 100.0))) * np.log(Pstart / P)
        T = T - 273.15  # back to °C
        return T, P"""

    # Dry adiabats
    Pstart = 1000.0  # hPa (reference starting pressure)
    for T0 in np.arange(-60, 441, 10):
        Tdry, Pdry = dry_adiabat(T0, Pstart)
        Ydry = np.log10(Pdry / 1000.0) if use_log10 else np.log(Pdry / 1000.0)
        Xdry = Tdry - (skew * Ydry)
        ax.plot(Xdry, Ydry, color=(0.6, 0.3, 0.0), linewidth=0.5, linestyle='--')

    # Moist adiabats
    for T0 in np.arange(-60, 51, 5):
        Tmoist, Pmoist = moist_adiabat(T0, Pstart)
        Ymoist = np.log10(Pmoist / 1000.0) if use_log10 else np.log(Pmoist / 1000.0)
        Xmoist = Tmoist - (skew * Ymoist)
        ax.plot(Xmoist, Ymoist, color=(0, 0.6, 0), linewidth=0.5, linestyle='--')

    # ---------------------------------------------------------------------------
    # Axis ticks and limits
    # ---------------------------------------------------------------------------
    #xticks = np.arange(-60, 15, 10) # Winter EOP
    xticks = np.arange(-60, 55, 10) # Winter + Summer EOP
    ax.set_xticks(xticks)
    ax.set_xticklabels([str(int(x)) for x in xticks])
    #ax.set_xlim([-65, 15]) # Winter EOP
    ax.set_xlim([-65, 55]) # Winter + Summer EOP
    ax.set_ylim([0, -1.65])

    ax.set_xlabel("Temperature at 1000 hPa (°C)")
    ax.set_ylabel("Pressure (hPa)")

    #ax.legend(loc="lower left", frameon=False)
    ax.legend(
        ['Temperature', 'Dewpoint'],
        loc='upper left',            # or 'northwest' equivalent
        #loc='lower left',           # or 'southwest' equivalent
        frameon=True,                # draw the box
        facecolor='white',           # white background
        edgecolor='black',           # black border
        framealpha=1.0,              # no transparency
        fancybox=True,               # rounded corners (optional)
    )
    ax.text(
        0.02, 0.02,
        "Note: Quicklook Use Only. \nDewpoint data may be inaccurate.",
        color='k',
        transform=ax.transAxes,
        fontsize=8,
        bbox=dict(
            boxstyle='round',      # rounded corners
            facecolor='white',     # background color
            edgecolor='black',     # border color
            alpha=1.0,             # opacity
            pad=0.3                # padding inside the box
        )
    )
    ax.grid(False)
    plt.tight_layout()

    # Build figure name
    start = datetime.datetime.strptime(f"{ds.attrs.get('startdate','')} {ds.attrs.get('starttime','')}", "%Y-%m-%d %H:%M:%S UTC")
    st = start.strftime("%Y%m%d_%H%M%S")
    figname = f"skewt_{st}_{ds.attrs.get('serial','')}.png"

    # Ensure the 'plots' subdirectory exists (create if needed)
    os.makedirs("plots", exist_ok=True)
    plot_path = os.path.join("plots", figname)
    plt.savefig(plot_path, dpi=450)
    print(f"Saved figure {figname} in directory {os.path.join(os.getcwd(), 'plots')}")

    plt.close()

    return

def main(args):
    # Keep track of created figures
    created_figures = []

    # Process given launches
    if args.files:
        for f in args.files:
            if os.path.isdir(f):
                 # Single directory
                for root, _, files in os.walk(f):
                    for file in files:
                        filepath = os.path.join(root, file)
                        # Processing radiosonde data from file
                        print(f"Processing file: {filepath}")
                        createskewt(filepath, args)

                        created_figures.append(filepath)
            else:
                # Processing radiosonde data from file
                print(f"Processing file: {f}")
                createskewt(f, args)

                created_figures.append(f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Radiosonde Skew-T Creator"
    )

    parser.add_argument(
        '-f', '--files',
        nargs='+',
        required=True,
        help='One or more local file paths or directories'
    )

    args = parser.parse_args()

    main(args)