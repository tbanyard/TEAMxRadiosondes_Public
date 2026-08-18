"""
Radiosonde Wind-Component Profile Plotter (multi-sonde overlay)
-----------------------------------------------------------------

Usage:
    python uvprofiles.py -f <file_or_directory> [<file_or_directory> ...] [options]

Example:
    python src/TEAMxRadiosondes/uvprofiles.py -f data/*.nc -l -b red : 1.5

Required arguments:
    -f, --files
        One or more file paths or directories containing radiosonde NetCDF data.
        Directories are walked recursively and every file found inside is
        treated as a sonde to plot. All sondes are drawn on a single figure.

Optional arguments:
    -b, --plot-border COLOR LINESTYLE LINEWIDTH
        Override the plot border appearance.
        Example: -b red : 1.5
        (Default border: k, -, 1)

    -l, --legend-title
        Add a title ("Sondes") above the filename legend.
        Off by default; enable by adding -l.

    -o, --outname
        Optional custom output filename (saved under ./plots/).
        Defaults to "uvprofiles_<N>sondes_<timestamp>.png".

Description:
    This script loads two or more radiosonde netCDF files and overlays their
    u/v wind-component profiles vs altitude on one figure. Each sonde gets
    its own shade of blue (u) and shade of red (v) so the profiles can be
    told apart, and sondes alternate between a black-edged marker and a
    borderless marker as a second visual cue. A compact legend (small font)
    identifies each sonde by its source filename.
"""

import os
import glob
import argparse

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib import ticker as mticker
from matplotlib.patches import FancyBboxPatch
from matplotlib.legend_handler import HandlerTuple

from TEAMxRadiosondes.funcs import lighten


def collect_files(paths):
    """Expand a mix of files/directories into a flat, sorted file list."""
    files = []
    for p in paths:
        if os.path.isdir(p):
            for root, _, fnames in os.walk(p):
                for fname in fnames:
                    files.append(os.path.join(root, fname))
        else:
            files.append(p)
    return sorted(files)


def plot_uv_alt_multi_sonde(files, args):
    """
    Overlay u/v wind-component profiles vs altitude for multiple sondes
    on a single figure, with a per-sonde filename legend.
    """

    n = len(files)
    if n == 0:
        print("No files to plot.")
        return None

    # Global font settings
    plt.rcParams.update({
        "axes.labelsize": 14,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 8,
        "axes.titlesize": 16,
    })

    config = {
        "MSIZE": 2,          # scatter marker size
        "BASE_U": "#4477AA",  # base zonal (u) colour -> shades of blue
        "BASE_V": "#EE6677",  # base meridional (v) colour -> shades of red
    }

    # Per-sonde lighten amounts: much darker at one end, much lighter at
    # the other, so overlapping profiles stay clearly separable even with
    # many sondes on the same figure.
    if n == 1:
        shade_amounts = [0.45]
    else:
        shade_amounts = np.linspace(0.02, 0.92, n)

    # Marker borders as a second visual cue: every sonde borderless except
    # the last one, which gets a black edge.
    edge_cycle = ["none"] * max(n - 1, 1) + ["black"]

    fig, ax = plt.subplots(figsize=(7.5, 12))

    handles_u, handles_v, labels = [], [], []
    zmax = 0.0

    for i, f in enumerate(files):
        try:
            ds = xr.load_dataset(f)
        except Exception as e:
            print(f"Skipping {f}: could not read ({e})")
            continue

        z = ds.alt.data
        u = ds.u.data
        v = ds.v.data

        mask = np.isfinite(z) & np.isfinite(u) & np.isfinite(v)
        z, u, v = z[mask], u[mask], v[mask]
        if z.size == 0:
            print(f"Skipping {f}: no valid data")
            continue

        sort_idx = np.argsort(z)
        z, u, v = z[sort_idx], u[sort_idx], v[sort_idx]
        z_km = z / 1000.0
        zmax = max(zmax, z_km.max())

        color_u = lighten(config["BASE_U"], shade_amounts[i])
        color_v = lighten(config["BASE_V"], shade_amounts[i])
        edgecolor = edge_cycle[i % len(edge_cycle)]
        edgelw = 0.2 if edgecolor == "black" else 0.0

        sc_u = ax.scatter(
            u, z_km, s=config["MSIZE"], color=color_u,
            edgecolors=edgecolor, linewidths=edgelw, alpha=0.85, zorder=3,
        )
        sc_v = ax.scatter(
            v, z_km, s=config["MSIZE"], color=color_v,
            edgecolors=edgecolor, linewidths=edgelw, alpha=0.85, zorder=3,
        )

        handles_u.append(sc_u)
        handles_v.append(sc_v)
        labels.append(os.path.basename(f))

    if zmax == 0.0:
        print("No plottable sondes found.")
        plt.close(fig)
        return None

    ax.set_ylim(0, zmax * 1.05)
    ax.axvline(0, color="black", lw=0.7, zorder=0.8)

    ax.set_xlabel("Wind component (m/s)")
    ax.set_ylabel("Altitude (km)")
    ax.set_title(f"Zonal (u) and Meridional (v) Wind Components")

    # Grid
    ax.set_axisbelow(True)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(5))
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(1))
    ax.xaxis.set_major_locator(mticker.MultipleLocator(5))
    ax.xaxis.set_minor_locator(mticker.MultipleLocator(1))
    ax.grid(which="major", ls="--", lw=0.7)
    ax.grid(which="minor", ls=":", lw=0.5)

    # Axis spine styling
    if args.plot_border:
        color, ls, lw = args.plot_border
        lw = float(lw)
    else:
        color, ls, lw = "k", "-", 1

    for spine in ax.spines.values():
        spine.set_edgecolor(color)
        spine.set_linestyle(ls)
        spine.set_linewidth(lw)

    # Per-sonde legend: each entry pairs the u-marker and v-marker for that
    # file so a single small-font line shows "filename -> (u shade, v shade)".
    leg = ax.legend(
        handles=list(zip(handles_u, handles_v)),
        labels=labels,
        handler_map={tuple: HandlerTuple(ndivide=None)},
        loc="upper left",
        frameon=False,
        title="Sondes" if args.legend_title else None,
        handletextpad=0.8,
        labelspacing=0.5,
        borderpad=0.8
    )

    # Dashed box around the legend, matching the single-sonde script's style
    fig.canvas.draw()
    bb = leg.get_window_extent(fig.canvas.get_renderer())
    bb_axes = bb.transformed(ax.transAxes.inverted())
    x0, y0 = bb_axes.x0, bb_axes.y0
    width, height = bb_axes.width, bb_axes.height

    ax.add_patch(
        FancyBboxPatch(
            (x0 - 0.005, y0 + 0.005),
            width,
            height,
            boxstyle="square,pad=0",
            linewidth=1.0,
            edgecolor="black",
            linestyle="--",
            facecolor="white",
            transform=ax.transAxes,
            zorder=2,
        )
    )
    leg.set_zorder(5)

    # Save figure
    if args.outname:
        figname = args.outname
    else:
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        figname = f"uvprofiles_{n}sondes_{timestamp}.png"

    os.makedirs("plots", exist_ok=True)
    plot_path = os.path.join("plots", figname)

    plt.savefig(plot_path, dpi=450, bbox_inches="tight", pad_inches=0.02)
    print(f"Saved figure {figname} in directory {os.path.join(os.getcwd(), 'plots')}")
    plt.close(fig)

    return fig


def main(args):
    files = collect_files(args.files)
    print(f"Found {len(files)} file(s) to plot:")
    for f in files:
        print(f"  {f}")
    plot_uv_alt_multi_sonde(files, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Radiosonde Wind-Component Profile Creator (multi-sonde overlay)"
    )

    parser.add_argument(
        "-f", "--files",
        nargs="+",
        required=True,
        help="One or more local file paths or directories, all overlaid on one figure"
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
        help="Add a 'Sondes' title above the filename legend."
    )

    parser.add_argument(
        "-o", "--outname",
        required=False,
        help="Optional custom output filename (saved under ./plots/)."
    )

    args = parser.parse_args()
    main(args)