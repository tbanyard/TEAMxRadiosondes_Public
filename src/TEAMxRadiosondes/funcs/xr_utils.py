"""
xr_utils.py

Utilities for cropping NetCDF files using XArray. Uses 'xr.sel' for subsetting.
"""

import os
import time
import psutil
import xarray as xr
from dask.diagnostics import ProgressBar

def get_ram_usage():
    """Returns the current RAM usage in GB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 3)  # Convert bytes to GB

def xr_clip(model_ds, run, sonde_tag, run_tag, lat_slice, lon_slice, time_slice):
    """
    Clips a model dataset using xarray's .sel() method based on track slices,
    profiles RAM usage, and saves/caches the result to disk.

    Parameters
    ----------
    model_ds : xr.Dataset
        The full model dataset to clip.
    run : ModelRun
        The model reader instance (e.g. MetUMRun) whose GRID_DIM_MAP defines
        which dataset dimensions correspond to lat/lon/time for this model.
    sonde_tag : str
        Identifier used to build a recognisable, collision-free temporary filename.
    lat_slice, lon_slice, time_slice : slice
        Bounding slices in model coordinate space, e.g. from ModelRun.get_sonde_geometry().

    Returns
    -------
    str
        Path to the clipped (or cached) output NetCDF file.
    """
    # Dynamically build sel() kwargs using only dimensions this model actually has,
    # via the model's own GRID_DIM_MAP - no hardcoded dimension names here.
    slice_by_axis = {"lat": lat_slice, "lon": lon_slice, "time": time_slice}
    sel_kwargs = {}
    for axis, dim_names in run.GRID_DIM_MAP.items():
        for dim_name in dim_names:
            if dim_name in model_ds.dims:
                sel_kwargs[dim_name] = slice_by_axis[axis]

    # Extract model tag and construct temporary path
    source = model_ds.encoding.get('source', 'unknown_source.nc')
    model_tag = os.path.basename(source).replace('.nc', '').split('_', 1)[-1]
    tmp_path = f"tmp_xrclipped_{sonde_tag}_{run_tag}_all_{model_tag}.nc"

    # Check if file already exists (cache hit)
    if os.path.exists(tmp_path):
        print(f"\nFound existing cached file: {tmp_path}. Skipping xr.sel clipping.")
        return tmp_path

    print("\nClipping model output to radiosonde track...")
    model_ds_clipped = model_ds.sel(**sel_kwargs)

    # Data load and RAM profiling
    t0 = time.time()
    ram_before = get_ram_usage()
    print(f"Current Process RAM: {ram_before:.2f} GB")

    print("\nLoading entire clipped dataset into memory...")
    with ProgressBar():
        model_ds_clipped = model_ds_clipped.load()

    t1 = time.time()
    ram_after = get_ram_usage()
    ds_size_gb = model_ds_clipped.nbytes / (1024 ** 3)

    print("Dataset successfully cached in RAM")
    print(f"  Time taken:               {t1 - t0:.2f}s")
    print(f"  Xarray Dataset Size:      {ds_size_gb:.2f} GB")
    print(f"  Physical RAM Before:      {ram_before:.2f} GB")
    print(f"  Physical RAM After:       {ram_after:.2f} GB")
    print(f"  Actual RAM Increase:     +{ram_after - ram_before:.2f} GB")
    
    # Save clipped dataset to a temporary netcdf file
    print(f"Saving clipped dataset to {tmp_path}...")
    model_ds_clipped.to_netcdf(tmp_path)

    return tmp_path