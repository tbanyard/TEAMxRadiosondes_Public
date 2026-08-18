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

    # Extract and assign MetPy units
    p = ds["pressure"].values * units.hPa
    t = (ds["temp"].values - 273.15) * units.degC
    td = (ds["dewp"].values - 273.15) * units.degC

    # Handle pressure units if they were in Pa
    if np.nanmax(p) > 2000 * units.hPa:
        p = p / 100.0

    # Setup the Figure and SkewT object
    fig = plt.figure(figsize=(9, 9))
    skew = SkewT(fig, rotation=45) # Standard 45-degree skew

    # Plot the data
    skew.plot(p, t, 'r', linewidth=2, label='Temperature')
    skew.plot(p, td, 'b', linewidth=2, label='Dewpoint')

    # Add standard background lines
    skew.plot_dry_adiabats(t0=np.arange(-60, 150, 10) * units.degC, 
                           color='brown', linewidth=0.5, alpha=0.5)
    
    # This fixes your moist adiabat "left-drift" issue automatically
    skew.plot_moist_adiabats(t0=np.arange(-40, 50, 5) * units.degC, 
                             color='green', linewidth=0.5, alpha=0.5)
    
    skew.plot_mixing_lines(color='black', linestyle=':', alpha=0.3)

    # Set axis limits and labels
    skew.ax.set_ylim(1000, 50)  # Pressure range
    skew.ax.set_xlim(-60, 40)   # Temperature range at 1000hPa
    
    plt.title(f"{ds.attrs.get('serial','')} {ds.attrs.get('startdate','')}", loc='left')
    plt.legend(loc='upper left', frameon=True)
    
    # Add disclaimer text box
    skew.ax.text(0.02, 0.02, "Note: Quicklook Use Only.\nDewpoint data may be unreliable.",
                 transform=skew.ax.transAxes, fontsize=8, bbox=dict(facecolor='white', alpha=1.0))

    # Save figure
    plt.savefig(f"plots/skewt_metpy_{ds.attrs.get('serial','')}.png", dpi=300)
    plt.close()

def createskewt(f, args):
    ds = xr.load_dataset(f)

    # --- Extract arrays ---
    # For skew-t
    P = ds["pressure"].values
    T = ds["temp"].values - 273.15 # In Celsius
    Td = ds["dewp"].values - 273.15 # In Celsius

    # For wind barbs
    u = ds["u"].values
    v = ds["v"].values
    z = ds["alt"].values

    # Extract launch time
    t = pd.to_datetime(ds.attrs["launch_time"])
    timestamp = t.strftime("%Y%m%d_%H%M%S")
    prettytimestamp = t.strftime("%d %b %Y %H:%M:%S UTC")

    # Sorting and masking
    mask = np.isfinite(z) & np.isfinite(u) & np.isfinite(v) & np.isfinite(P) & np.isfinite(T) & np.isfinite(Td)
    z, u, v, P, T, Td = z[mask], u[mask], v[mask], P[mask], T[mask], Td[mask]
    sort_idx = np.argsort(z)
    z, u, v, P, T, Td = z[sort_idx], u[sort_idx], v[sort_idx], P[sort_idx], T[sort_idx], Td[sort_idx]
    z_km = z / 1000.0

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
    fig = plt.figure(figsize=(7, 7))
    gs = fig.add_gridspec(1, 2, width_ratios=[4, 0.35], wspace=0.05)

    ax = fig.add_subplot(gs[0, 0])
    ax_barb = fig.add_subplot(gs[0, 1])
    ax.set_title(
        #f"Skew-T Diagram\n"
        f"{ds.attrs['site_name']}: {ds.attrs['serial']} ({ds.attrs.get('source','')})\n"
        f"{t:%d %b %Y %H:%M:%S UTC}",
        fontsize = 10
    )

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
    # Mixing ratio lines
    # ---------------------------------------------------------------------------
    w_values = np.array([0.04, 0.1, 0.2, 0.4, 1, 2, 4, 8, 16, 32])  # g/kg

    Pmix = np.linspace(1000, 100, 80)  # hPa
    Ymix = np.log10(Pmix / 1000.0) if use_log10 else np.log(Pmix / 1000.0)

    for w in w_values:
        w_kgkg = w / 1000.0  # convert g/kg → kg/kg

        # Compute vapour pressure from mixing ratio
        e = (w_kgkg * Pmix) / (0.622 + w_kgkg)

        # Invert Bolton formula to get temperature
        ln_ratio = np.log(e / 6.112)
        Tmix = (243.5 * ln_ratio) / (17.67 - ln_ratio)  # °C

        Xmix = Tmix - (skew * Ymix)

        ax.plot(Xmix, Ymix,
                color='red',
                linestyle=':',
                linewidth=0.5,
                alpha=0.75)

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
    
    # Mixing ratio labels
    x_positions = [0.128, 0.183, 0.24, 0.3, 0.385, 0.46, 0.544, 0.628, 0.71, 0.8, 0.97]   # ← you’ll extend this
    w_labels = ["0.04", "0.1", "0.2", "0.4", "1.0", "2.0", "4.0", "8.0", "16", "32", "g/kg"]

    for x, lab in zip(x_positions, w_labels):
        ax.text(
            x, -0.013,
            lab,
            color='red',
            transform=ax.transAxes,
            fontsize=5
        )

    ax.grid(False)

    # Wind barbs
    z_floor = np.floor(z_km)
    km_bins = np.arange(int(z_floor.min()), int(z_floor.max()) + 1)

    barb_p, barb_u, barb_v = [], [], []

    for km in km_bins[1:]:
        m_layer = (z_floor == km)
        if np.sum(m_layer) == 0:
            continue

        barb_u.append(np.mean(u[m_layer]))
        barb_v.append(np.mean(v[m_layer]))

        # Use median pressure in that layer
        barb_p.append(np.median(P[m_layer]))

    barb_p = np.array(barb_p)
    barb_u = np.array(barb_u)
    barb_v = np.array(barb_v)

    Ybarb = np.log10(barb_p / 1000.0) if use_log10 else np.log(barb_p / 1000.0)

    # ---------------------------------------------------------------------------
    # Wind barbs
    # ---------------------------------------------------------------------------
    ax_barb.set_facecolor("none")
    ax_barb.set_xticks([])
    ax_barb.set_yticks([])
    ax_barb.tick_params(left=False, right=False, top=False, bottom=False)
    ax_barb.set_frame_on(False)

    # Match skew-T vertical scale
    ax_barb.set_ylim(ax.get_ylim())
    ax_barb.set_xlim(0, 1)

    if len(Ybarb) > 0:
        ax_barb.barbs(
            np.full_like(Ybarb, 0.6),
            Ybarb,
            barb_u,
            barb_v,
            length=5.4,
            linewidth=0.8,
            barb_increments=dict(half=2.5, full=5, flag=25),
            color="black"
        )

    plt.tight_layout()

    # Build figure name
    figname = f"{timestamp}_{ds.attrs.get('serial','')}_skewt.png"

    # Ensure the 'plots' subdirectory exists (create if needed)
    os.makedirs("plots", exist_ok=True)
    plot_path = os.path.join("plots", figname)
    plt.savefig(plot_path, dpi=175, bbox_inches='tight', pad_inches=0.02)
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