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
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.patches import RegularPolygon, Polygon
from scipy.signal import savgol_filter

def createhodograph(f, args):
    ds = xr.load_dataset(f)

    # Configuration
    config = {
        "N_ARROWS": 30,  # total number of arrows across whole plot
        "SMOOTH_LOW": 155,  # lower window (smaller = keeps more high-freq)
        "SMOOTH_HIGH": 801,  # higher window (larger = defines background)
        "SMOOTH_POLY": 2,
        "TRI_SIZE_FACTOR": 0.01,  # relative triangle size
        "EDGE_LW": 0.5,  # thin black outline on triangles
        "COLORMAP": cm.get_cmap('twilight_shifted'),
        "ALT_MIN": 0.0,
        "ALT_MAX": 30.0,
    }

    # Extract arrays
    u = ds.u.data
    v = ds.v.data
    alt = ds.alt.data
    alt_km = alt / 1000.0

    # Filter by altitude range
    mask = (alt_km >= config['ALT_MIN']) & (alt_km <= config['ALT_MAX'])

    u = np.where(mask, u, np.nan)
    v = np.where(mask, v, np.nan)
    alt_km = np.where(mask, alt_km, np.nan)

    # Split into continuous segments (No NaNs)
    coords = np.column_stack((u, v, alt_km))
    valid = ~np.isnan(coords).any(axis=1)

    segments = []
    if np.any(valid):
        # Find transitions from regions with data to those without
        changes = np.where(np.diff(valid.astype(int)) != 0)[0] + 1
        # Build an array of segment boundaries
        indices = np.r_[0, changes, len(valid)]
        for i in range(len(indices) - 1):
            seg = coords[indices[i]:indices[i + 1]]
            # Only use a segment if it contains no NaNs and it is larger than
            # SMOOTH_HIGH for perturbation hodographs and 3 for background hodographs
            if args.hodograph_type == "background":
                minseglength = 3
            elif args.hodograph_type == "perturbation":
                minseglength = config["SMOOTH_HIGH"]        
            if np.all(~np.isnan(seg)) and len(seg) > minseglength:
                segments.append(seg)

    # Create figure
    fig, ax = plt.subplots(figsize=(7, 7))
    norm = Normalize(vmin=np.nanmin(alt_km), vmax=np.nanmax(alt_km))

    # Determine plot scale for sizing triangles
    rng = max(np.ptp(u[np.isfinite(u)]), np.ptp(v[np.isfinite(v)]), 1.0)
    tri_radius = config['TRI_SIZE_FACTOR'] * rng

    # Smooth and plot each segment, computing SG bandpass if running perturbation toggle
    all_u, all_v, all_alt = [], [], []
    for seg in segments:
        u_seg, v_seg, alt_seg = seg.T

        if args.hodograph_type == "background":
            # Smooth the data
            if len(u_seg) >= config['SMOOTH_LOW']:
                u_sm = savgol_filter(u_seg, config['SMOOTH_LOW'], config['SMOOTH_POLY'])
                v_sm = savgol_filter(v_seg, config['SMOOTH_LOW'], config['SMOOTH_POLY'])
            else:
                u_sm, v_sm = u_seg, v_seg

            # Store combined arrays for later arrow placement
            all_u.extend(u_sm)
            all_v.extend(v_sm)

        elif args.hodograph_type == "perturbation":
            # Smooth the data twice (high-pass + low-pass)
            u_low = savgol_filter(u_seg, config['SMOOTH_LOW'], config['SMOOTH_POLY'])
            v_low = savgol_filter(v_seg, config['SMOOTH_LOW'], config['SMOOTH_POLY'])
            u_high = savgol_filter(u_seg, config['SMOOTH_HIGH'], config['SMOOTH_POLY'])
            v_high = savgol_filter(v_seg, config['SMOOTH_HIGH'], config['SMOOTH_POLY'])

            # Bandpass filter: remove slow background and noise
            u_bp = u_low - u_high
            v_bp = v_low - v_high

            # Store combined arrays for later arrow placement
            all_u.extend(u_bp)
            all_v.extend(v_bp)

        all_alt.extend(alt_seg)

        # Plot coloured line for background hodograph
        if args.hodograph_type == "background":
            for i in range(len(u_sm) - 1):
                mid_alt = 0.5 * (alt_seg[i] + alt_seg[i + 1])
                ax.plot(u_sm[i:i + 2], v_sm[i:i + 2],
                        color=config['COLORMAP'](norm(mid_alt)), lw=2.2)
        
        # Plot coloured line for perturbation hodograph
        elif args.hodograph_type == "perturbation":
            for i in range(len(u_bp) - 1):
                mid_alt = 0.5 * (alt_seg[i] + alt_seg[i + 1])
                ax.plot(u_bp[i:i + 2], v_bp[i:i + 2],
                        color=config['COLORMAP'](norm(mid_alt)), lw=2.2)
            
    # Add evenly spaced arrowheads along entire path
    all_u = np.array(all_u)
    all_v = np.array(all_v)
    all_alt = np.array(all_alt)

    # Add evenly spaced, properly oriented isosceles triangles
    if len(all_u) > 2:
        if len(all_u) > config['N_ARROWS']:
            idx = np.linspace(0, len(all_u) - 2, config['N_ARROWS'], dtype=int)
        else:
            idx = np.arange(len(all_u) - 1)

        for i in idx:
            dx = all_u[i + 1] - all_u[i]
            dy = all_v[i + 1] - all_v[i]
            if dx == 0 and dy == 0:
                continue

            angle = np.arctan2(dy, dx)
            x0 = 0.5 * (all_u[i] + all_u[i + 1])
            y0 = 0.5 * (all_v[i] + all_v[i + 1])
            mid_alt = 0.5 * (all_alt[i] + all_alt[i + 1])
            color = config['COLORMAP'](norm(mid_alt))

            # Forward offset
            offset = 0.5 * tri_radius
            x0 += offset * np.cos(angle)
            y0 += offset * np.sin(angle)

            # Define isosceles triangle geometry
            tip_len = 1.6 * tri_radius     # how long the tip is
            base_half = 0.7 * tri_radius   # half-width of the base

            # Vertices (tip, left base, right base)
            tip_x = x0 + tip_len * np.cos(angle)
            tip_y = y0 + tip_len * np.sin(angle)
            left_x = x0 - base_half * np.cos(angle) + base_half * np.sin(angle)
            left_y = y0 - base_half * np.sin(angle) - base_half * np.cos(angle)
            right_x = x0 - base_half * np.cos(angle) - base_half * np.sin(angle)
            right_y = y0 - base_half * np.sin(angle) + base_half * np.cos(angle)

            verts = [(tip_x, tip_y), (left_x, left_y), (right_x, right_y)]

            tri = Polygon(
                verts,
                closed=True,
                facecolor=color,
                edgecolor='k',
                linewidth=config['EDGE_LW'],
                alpha=0.95,
                zorder=6
            )
            ax.add_patch(tri)

    if args.hodograph_type == "perturbation":
        # Centered limits (same range on both axes)
        try:
            max_range = np.max([
                np.nanmax(np.abs(all_u)),
                np.nanmax(np.abs(all_v))
            ])
            ax.set_xlim(-max_range * 1.1, max_range * 1.1)
            ax.set_ylim(-max_range * 1.1, max_range * 1.1)
        except:
            print(f"Plot centering failed for {f}")

    # Colorbar and aesthetics
    sm = cm.ScalarMappable(norm=norm, cmap=config['COLORMAP'])
    sm.set_array(all_alt)
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label('Altitude (km)')

    ax.set_aspect('equal', 'box')
    ax.axhline(0, color='gray', lw=0.6, ls='--')
    ax.axvline(0, color='gray', lw=0.6, ls='--')
    t = pd.to_datetime(ds.attrs["launch_time"])
    timestamp = t.strftime("%Y%m%d_%H%M%S")

    if args.hodograph_type == "background":
        ax.set_xlabel('u (m/s)')
        ax.set_ylabel('v (m/s)')
        ax.set_title(
            f"{args.hodograph_type.capitalize()} Hodograph\n"
            f"{ds.attrs['site_name']}: {ds.attrs['serial']} ({ds.attrs.get('source','')})\n"
            f"{t:%d %b %Y %H:%M:%S UTC}",
            fontsize = 10
        )
        #ax.set_title(f"{ds.attrs.get('site_name', '')}: {ds.attrs.get('serial', '')} — u vs v (colored by altitude)\n{t:%d %b %Y %H:%M UTC}")
        figname = f"{timestamp}_{ds.attrs.get('serial','')}_hodograph.png"
    elif args.hodograph_type == "perturbation":
        ax.set_xlabel("u' (m/s)")
        ax.set_ylabel("v' (m/s)")
        ax.set_title(
            f"{args.hodograph_type.capitalize()} Hodograph"
            f"{ds.attrs['site_name']}: {ds.attrs['serial']} ({ds.attrs.get('source','')})\n"
            f"{t:%d %b %Y %H:%M:%S UTC}",
            fontsize = 10
        )
        #ax.set_title(f"Perturbation Hodograph — bandpassed u′ vs v′ (SG filter)\n{t:%d %b %Y %H:%M UTC}")
        figname = f"{timestamp}_{ds.attrs.get('serial','')}_hodograph_perts.png"
    
    ax.grid(True, ls=':', lw=0.5)
    plt.tight_layout()

    # Ensure the 'plots' subdirectory exists (create if needed)
    os.makedirs("plots", exist_ok=True)
    plot_path = os.path.join("plots", figname)
    plt.savefig(plot_path, dpi=200, bbox_inches='tight', pad_inches=0.02)
    print(f"Saved figure {figname} in directory {os.path.join(os.getcwd(), 'plots')}")
    plt.close()

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
                        createhodograph(filepath, args)

                        created_figures.append(filepath)
            else:
                # Processing radiosonde data from file
                print(f"Processing file: {f}")
                createhodograph(f, args)

                created_figures.append(f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Radiosonde Hodograph Creator"
    )

    parser.add_argument(
        '-f', '--files',
        nargs='+',
        required=True,
        help='One or more local file paths or directories'
    )

    parser.add_argument(
        '-t', '--hodograph-type',
        required=True,
        choices=['background', 'perturbation'],
        help='Choose background or perturbation hodograph'
    )

    args = parser.parse_args()

    main(args)