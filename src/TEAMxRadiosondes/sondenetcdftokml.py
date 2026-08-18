import os
import sys
import sondehub
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import pandas as pd
import datetime
import time
import argparse
from pathlib import Path
import simplekml
import random
from matplotlib import cm

# Define directories
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, '..', '..', 'data')

def createkmlfromnetcdf(filepath, args):
    ds = xr.open_dataset(filepath)
    lon = ds.lon.data
    lat = ds.lat.data
    alt = ds.alt.data

    # Sort out NaNs with fill values
    lat = np.where(lat > -1e10, lat, np.nan)
    lon = np.where(lon > -1e10, lon, np.nan)
    alt = np.where(alt > -1e10, alt, np.nan)

    site_name = getattr(ds, "site_name", "Unknown")
    serial = getattr(ds, "serial", "Unknown")

    # Prefer launch_time, else use startdate + starttime
    if hasattr(ds, "launch_time"):
        time_str = ds.launch_time
        # Example: "2025-02-20T10:58:00+00:00"
        launch_dt = datetime.datetime.fromisoformat(time_str.replace("Z", "+00:00"))
    elif hasattr(ds, "startdate") and hasattr(ds, "starttime"):
        date_str = ds.startdate
        time_str = ds.starttime.split()[0]  # remove possible "UTC"
        launch_dt = datetime.datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M:%S")
    else:
        launch_dt = datetime.datetime.utcnow()  # fallback

    # Format filename
    site_name_str = site_name.upper()
    site_name_str = site_name_str.ljust(8, "_") # Pad to 8 characters for consistent filename formatting
    filename = f"{launch_dt.strftime('%Y%m%d_%H%M%S')}_{serial}_{site_name_str}_SONDE.kml"

    # Create KML object
    kml = simplekml.Kml()

    # Pick a color from a colormap
    """colormap = cm.get_cmap('viridis', 10)
    rgb = colormap(random.randint(0, 9))[:3]
    line_color = simplekml.Color.rgb(
        int(rgb[0]*255),
        int(rgb[1]*255),
        int(rgb[2]*255)
    )"""

    colormap = cm.get_cmap('tab10')
    rgb = colormap(random.random())[:3]

    line_color = simplekml.Color.rgb(
        int(rgb[0]*255),
        int(rgb[1]*255),
        int(rgb[2]*255)
    )

    # Combine into one array for convenience
    coords = np.column_stack((lon, lat, alt))

    # Identify valid (non-NaN) indices
    valid = ~np.isnan(coords).any(axis=1)

    # Find breaks where valid → invalid or vice versa
    segments = []
    if np.any(valid):
        # Find indices where validity changes
        changes = np.where(np.diff(valid.astype(int)) != 0)[0] + 1
        indices = np.r_[0, changes, len(valid)]

        # Build segments
        for i in range(len(indices)-1):
            segment = coords[indices[i]:indices[i+1]]
            if np.all(~np.isnan(segment)):
                segments.append(segment)

    # Create a separate line for each continuous segment
    for i, segment in enumerate(segments):
        linestring = kml.newlinestring(name=f"Radiosonde Path {i+1}")
        linestring.coords = [tuple(x) for x in segment]
        linestring.altitudemode = simplekml.AltitudeMode.absolute
        linestring.style.linestyle.width = 2.5
        linestring.style.linestyle.color = line_color

    # Save to file
    os.chdir(data_dir)
    kml.save(filename)
    os.chdir('..')
    print(f'Created file {filename} in {os.path.abspath(data_dir)}')

def main(args):
    # Keep track of created files
    created_files = []

    # Process given files
    if args.files:
        for f in args.files:
            if os.path.isdir(f):
                # Single directory
                for root, _, files in os.walk(f):
                    for file in files:
                        filepath = os.path.join(root, file)
                        # Processing radiosonde data from file
                        print(f"Processing file: {filepath}")
                        createkmlfromnetcdf(filepath, args)

                        created_files.append(filepath)
            else:
                # Processing radiosonde data from file
                print(f"Processing file: {f}")
                createkmlfromnetcdf(f, args)

                created_files.append(f)
    
    # Success message depending on how many were processed
    if len(created_files) == 1:
        print("Radiosonde KML file created successfully.")
    elif len(created_files) > 1:
        print("Radiosonde KML files created successfully.")
    else:
        print("No KML files created.") # Should not normally happen

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Radiosonde KML File Creator"
    )

    group = parser.add_mutually_exclusive_group(required=True)

    group.add_argument(
        '-f', '--files',
        nargs='+',
        help='One or more local radiosonde netcdf file paths or directories'
    )

    args = parser.parse_args()

    main(args)