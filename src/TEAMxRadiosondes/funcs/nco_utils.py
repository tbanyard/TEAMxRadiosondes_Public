"""
nco_utils.py

Utilities for cropping NetCDF files using NCO (NetCDF Operators) command-line tools. Uses 'ncks' for subsetting.
"""

import os
import time
import xarray as xr
import subprocess

def crop_file_with_ncks(path, tmp_out, dim_bounds):
    """Crop a single NetCDF file to the given dimension bounds using ncks.

    Parameters
    ----------
    path : str
        Path to the input NetCDF file.
    tmp_out : str
        Path to write the cropped output file to. If this already exists,
        the file is assumed to be a valid cached result and is not regenerated.
    dim_bounds : dict
        Mapping of dimension name -> (min_index, max_index), e.g. as returned by
        ModelRun.get_dim_bounds(). Dimensions not present in this specific file,
        or with a value of None, are skipped.

    Returns
    -------
    bool
        True if a new cropped file was created, False if an existing cached
        file was found and reused.
    """
    if os.path.exists(tmp_out):
        return False

    # Peek at the file to see which dimensions it actually contains
    with xr.open_dataset(path, chunks={}) as f_meta:
        active_dims = f_meta.dims

    # Build the ncks command using only dimensions present in this file
    ncks_cmd = ["ncks"]
    for dim_name, bounds in dim_bounds.items():
        if dim_name in active_dims and bounds is not None:
            ncks_cmd.append(f"-d")
            ncks_cmd.append(f"{dim_name},{bounds[0]},{bounds[1]}")
    ncks_cmd.extend([path, tmp_out])

    subprocess.run(ncks_cmd, check=True)
    return True


def crop_files_with_ncks(files_dict, dim_bounds, sonde_tag, run_tag):
    """Crop every file in files_dict to dim_bounds via ncks, with progress reporting.

    Parameters
    ----------
    files_dict : dict
        Mapping of var_name -> file path, e.g. as returned by ModelRun.discover_files().
    dim_bounds : dict
        Mapping of dimension name -> (min_index, max_index), e.g. as returned by
        ModelRun.get_dim_bounds().
    sonde_tag : str
        Identifier used to build recognisable, collision-free temporary filenames.

    Returns
    -------
    list of str
        Paths to the cropped (or cached) output files, in the same order as files_dict.
    """
    print("\nLaunching NCO cropping tools in the background if necessary...")
    tmp_output_files = []

    for idx, (var_name, path) in enumerate(files_dict.items()):
        t_file = time.time()
        filename = os.path.basename(path)

        # Use var_name (e.g., 'u', 'sh') to make a clean, recognizable temporary filename
        tmp_out = f"tmp_subset_{sonde_tag}_{run_tag}_{filename}"
        tmp_output_files.append(tmp_out)

        print(f"  Processing variable [{var_name}] ({idx + 1}/{len(files_dict)}): {filename}")

        was_created = crop_file_with_ncks(path, tmp_out, dim_bounds)

        if was_created:
            print(f"  -> Cropped successfully in {time.time() - t_file:.1f}s")
        else:
            print(f"  [{var_name}] Found existing cached file: {tmp_out}. Skipping file save.")

    return tmp_output_files