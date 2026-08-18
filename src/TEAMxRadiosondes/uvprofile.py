"""
Radiosonde Wind-Component Profile Plotter
-----------------------------------------

Usage:
    python uvprofile.py -f <file_or_directory> [options]

Example:
    python src/TEAMxRadiosondes/uvprofile.py -f data/202502261702R5040846_Nph_all.nc -l -b red : 1.5

Required arguments:
    -f, --files
        One or more file paths or directories containing radiosonde NetCDF data.

Optional arguments:
    -b, --plot-border COLOR LINESTYLE LINEWIDTH
        Override the plot border appearance.
        Example: -b red : 1.5
        (Default border: red, :, 1.5)

    -l, --legend-title
        Toggle the "Barb Legend" title on the wind-barb legend.
        Off by default; enable by adding -l.

Description:
    This script loads radiosonde data from netCDF
    and generates a wind-component profile plot,
    with optional custom border styling
    and an optional barb legend title display.
"""

import os
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse
from matplotlib import ticker as mticker
from matplotlib.patches import FancyBboxPatch
from matplotlib.gridspec import GridSpec

from TEAMxRadiosondes.funcs import lighten

def plot_uv_alt_single_sonde(f, args):
    """
    Plot u/v components vs altitude + side wind-barbs + compact legend.
    """

    # Global Font Settings
    plt.rcParams.update({
        "axes.labelsize": 14,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
        "axes.titlesize": 17,

    })

    # Load data
    ds = xr.load_dataset(f)

    # Configuration
    config = {
        "MSIZE": 30, # Size of scatter markers
        "BASE_U": "#4477AA", # Zonal wind marker colour
        "BASE_V": "#EE6677", # Meridional wind marker colour
        "EDGECOLORS": "k", # Marker edge colour
    }

    # Initialise figure
    fig = plt.figure(figsize=(7, 13))
    gs = GridSpec(
        2, 2,
        height_ratios=[15, 1],
        width_ratios=[4, 0.8],
        hspace=0.25,
        wspace=0.0,
        figure=fig
    )

    ax = fig.add_subplot(gs[0, 0]) # main panel
    ax_barb = fig.add_subplot(gs[0, 1]) # barb column
    ax_leg = fig.add_axes([0.17, 0.16, 0.90, 0.12]) # wind-barb legend panel

    # Colours
    color_u = lighten(config['BASE_U'], 0.55)
    color_v = lighten(config['BASE_V'], 0.55)

    # Extract data
    z = ds.alt.data
    u = ds.u.data
    v = ds.v.data

    mask = np.isfinite(z) & np.isfinite(u) & np.isfinite(v)
    z, u, v = z[mask], u[mask], v[mask]

    sort_idx = np.argsort(z)
    z, u, v = z[sort_idx], u[sort_idx], v[sort_idx]

    z_km = z / 1000.0

    # Extract timestamp for filenaming
    t = pd.to_datetime(ds.attrs["launch_time"])
    timestamp = t.strftime("%Y%m%d_%H%M%S")

    # Plot points for profiles
    ax.scatter(u, z_km, s=config['MSIZE'], color=color_u,
               edgecolors=config['EDGECOLORS'], linewidths=0.1, alpha=0.9, label="u")
    ax.scatter(v, z_km, s=config['MSIZE'], color=color_v,
               edgecolors=config['EDGECOLORS'], linewidths=0.1, alpha=0.9, label="v")

    ax.set_ylim(0, z_km.max() * 1.05)
    ax.axvline(0, color="black", lw=0.7, zorder=0.8)

    ax.set_xlabel("Wind component (m/s)")
    ax.set_ylabel("Altitude (km)")
    ax.set_title(
        #f"Zonal (u) and Meridional (v) Wind Components\n"
        f"{ds.attrs['site_name']}: {ds.attrs['serial']} ({ds.attrs.get('source','')})\n"
        f"{t:%d %b %Y %H:%M:%S UTC}"
    )

    # Grid settings
    ax.set_axisbelow(True)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(5))
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(1))
    ax.xaxis.set_major_locator(mticker.MultipleLocator(5))
    ax.xaxis.set_minor_locator(mticker.MultipleLocator(1))
    ax.grid(which="major", ls='--', lw=0.7)
    ax.grid(which="minor", ls=':', lw=0.5)

    # Set axis spine styling
    if args.plot_border:
        color, ls, lw = args.plot_border
        lw = float(lw)
    else:
        color, ls, lw = "k", "-", 1
        #color, ls, lw = "red", ":", 1.5 # Make axis spined red and dotted

    for spine in ax.spines.values():
        spine.set_edgecolor(color)
        spine.set_linestyle(ls)
        spine.set_linewidth(lw)

    # Sort out profile legend
    leg = ax.legend(loc="upper left", frameon=False)
    bb = leg.get_window_extent(ax.figure.canvas.get_renderer())
    bb_axes = bb.transformed(ax.transAxes.inverted())

    # Extract dimensions of legend in axes coordinates
    x0, y0 = bb_axes.x0, bb_axes.y0
    width, height = bb_axes.width, bb_axes.height

    # Add a dashed box around the legend
    ax.add_patch(
        FancyBboxPatch(
            (x0-0.005, y0+0.005),
            width,
            height,
            boxstyle="square,pad=0",
            linewidth=1.0,
            edgecolor="black",
            linestyle="--",
            facecolor="white",
            transform=ax.transAxes,
            zorder=5,
        )
    )

    # Plot wind barbs
    ax_barb.set_facecolor("none")
    ax_barb.set_xticks([])
    ax_barb.set_yticks([])
    ax_barb.tick_params(left=False, right=False, top=False, bottom=False, which='both')
    ax_barb.set_frame_on(False)

    z_floor = np.floor(z_km)
    km_bins = np.arange(int(z_floor.min()), int(z_floor.max()) + 1)

    barb_z, barb_u, barb_v = [], [], []

    for km in km_bins[1:]:
        m_layer = (z_floor == km)
        if np.sum(m_layer) == 0:
            continue
        barb_z.append(km + 0.5)
        barb_u.append(np.mean(u[m_layer]))
        barb_v.append(np.mean(v[m_layer]))

    barb_z = np.array(barb_z)
    barb_u = np.array(barb_u)
    barb_v = np.array(barb_v)

    ax_barb.set_ylim(ax.get_ylim())
    ax_barb.set_xlim(0, 1)

    if len(barb_z) > 0:
        ax_barb.barbs(
            np.full_like(barb_z, 0.7),
            barb_z,
            barb_u,
            barb_v,
            length=5.5,
            linewidth=1.0,
            barb_increments=dict(half=2.5, full=5, flag=25),
            color="black"
        )

    # Sort out wind barb legend
    ax_leg.set_axis_off()
    ax_leg.set_xlim(0, 1)
    ax_leg.set_ylim(0, 1)

    x0 = 0.802
    y0 = 0.52

    # Font sizes
    title_fs = 14   # for "65 kts"
    mid_fs   = 10   # for "50 10 5"

    # Add example barb
    ax_leg.barbs(
        [x0], [y0],
        [33], [0],
        length=9,
        barb_increments=dict(half=2.5, full=5, flag=25),
        color="black"
    )

    # Text labels
    ax_leg.text(x0 - 0.11, y0 - 0.1, "50 10 5", fontsize=mid_fs)
    ax_leg.text(x0 + 0.006, y0 + 0.15, "65 kts", ha="center", va="top", fontsize=title_fs)
    if args.legend_title:
        ax_leg.text(x0 - 0.0275, y0 + 0.34, "Barb Legend",
                    ha="center", va="top", fontsize=title_fs)

    # Dashed box around legend
    ax_leg.add_patch(
        FancyBboxPatch(
            (x0 - 0.117, y0 - 0.135),
            0.177, 0.33,
            boxstyle="square,pad=0",
            linewidth=1.0,
            edgecolor="black",
            linestyle="--",
            facecolor="none",
            zorder=5,
        )
    )

    # Save figure
    figname = f"{timestamp}_{ds.attrs['serial']}_uvprofile.png"
    os.makedirs("plots", exist_ok=True)
    plot_path = os.path.join("plots", figname)

    plt.savefig(plot_path, dpi=175, bbox_inches='tight', pad_inches=0.02)
    print(f"Saved figure {figname} in directory {os.path.join(os.getcwd(), 'plots')}")
    plt.close()

    return fig


def main(args):
    if args.files:
        for f in args.files:
            if os.path.isdir(f):
                for root, _, files in os.walk(f):
                    for file in files:
                        filepath = os.path.join(root, file)
                        print(f"Processing file: {filepath}")
                        plot_uv_alt_single_sonde(filepath, args)
            else:
                print(f"Processing file: {f}")
                plot_uv_alt_single_sonde(f, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Radiosonde Wind-component Profile Creator"
    )

    parser.add_argument(
        "-f", "--files",
        nargs="+",
        required=True,
        help="One or more local file paths or directories"
    )

    parser.add_argument(
        "-b", "--plot-border",
        nargs=3,
        metavar=("COLOR", "LINESTYLE", "LINEWIDTH"),
        help="Border styling: COLOR LINESTYLE LINEWIDTH, e.g. -b red : 1.5",
        required=False
    )

    parser.add_argument(
        "-l", "--legend-title",
        action="store_true",
        help="Toggle the 'Barb Legend' title on or off."
    )

    args = parser.parse_args()
    main(args)
