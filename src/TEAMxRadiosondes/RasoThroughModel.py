"""
Interpolate radiosonde track onto any model field.
"""

# ======= #
# IMPORTS #
# ======= #

import os
import glob
import argparse
import time
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.feature import NaturalEarthFeature
from cartopy.io.shapereader import natural_earth, Reader
from dask.diagnostics import ProgressBar
import datetime
import re

from TEAMxRadiosondes.funcs import nco_utils, xr_utils, haversine, read_config_file, find_nearest_station, set_site_name, buildxrds

# ============================================================= #
#                         MODEL READERS                         #
# ============================================================= #
# Each model is given its own class describing:                 #
#   - the model output file syntax       (discover_files)       #
#   - how to find the main data variable (find_main_variable)   #
#   - how to find lat/lon/height coord names (find_coord_names) #
#                                                               #
# Everything else is shared in the class ModelRun.              #
# To add support for a new model, write a new ModelRun subclass #
# which implements all model-specific code, then register it in #
# MODEL_READERS before setting CONFIG["model"] accordingly      #
#                                                               #
# ============================================================= #

class ModelRun:
    # Name of model
    MODEL_NAME = "Unknown Model"
    
    # Default simple model grid dimensions, this is usually overriden by subclasses
    GRID_DIM_MAP = {
        "lon": ["longitude"],
        "lat": ["latitude"],
        "time": ["time"],
    }
    
    """Base class to read the config file and output basic operations."""
    def __init__(self, config):
        self.config = config
        self.directory = config["directory"]
        self.available_files = self.discover_files() # Use discover_files from subclass
        self._open_datasets = {}  # Create cache of opened datasets (currently unused)
        self.pole_lat = 90.0 # default for unrotated grid (i.e. true north pole)
        self.pole_lon = 0.0

    def discover_files(self):
        """This is a subclass function and should not run in the main class"""
        raise NotImplementedError

    def get_dataset(self):
        """"Lazily open the datasets for a given file prefix.
        Returns None if no matching file was found.
        """
        print("available files", self.available_files)

        # Open all datasets into a list using xarray.open_dataset via the helper
        # function _xropen_single, which itself appropriately chunks the files using 
        # _safely_chunk_var if required
        datasets = [self._xropen_single(path) for path in self.available_files.values()]
        combined = self._combine_datasets(datasets)
        self.find_pole_coords(combined)
        return combined

    def _combine_datasets(self, datasets):
        """In the general case, e.g. for MetUM, just merge by variable name.
           This will be overidden by specific model subclasses, e.g. AROME"""
        return self._merge_by_var(datasets)
    
    def _merge_by_var(self, datasets):
        """Combines datasets containing different variables for the same model run"""
        # Save the first dataset as its own ds
        ds = datasets[0].copy()
        # Manually loop through the remaining datasets and inject missing variables
        for next_ds in datasets[1:]:
            for var_name in next_ds.data_vars:
                if var_name not in ds:
                    ds[var_name] = next_ds[var_name]
        return ds

    def find_main_variable(self, ds):
        """This is also a subclass function and should not run in the main class"""
        raise NotImplementedError
    
    def find_coord_names(self, ds):
        """This is also a subclass function and should not run in the main class"""
        raise NotImplementedError

    def find_pole_coords(self, ds):
        """Default: no rotated pole. Subclasses override if their grid uses one."""
        return self.pole_lat, self.pole_lon

    @property
    def model_crs(self):
        """Rotated-pole CRS for this model run (identity transform if unrotated).
        N.B. The @property decorator turns this method into a getter allowing it
        to be accessed as if it were an attribute, i.e. without () at the end."""
        return ccrs.RotatedPole(pole_longitude=self.pole_lon, pole_latitude=self.pole_lat)

    def transform_lonlat(self, lon, lat):
        """Transform real-world lon/lat arrays into model grid coordinates."""
        real_world_crs = ccrs.PlateCarree()
        transformed = self.model_crs.transform_points(real_world_crs, lon, lat)
        return transformed[:, 0], transformed[:, 1]

    def wrap_longitudes(self, lon, ds):
        """Default: no wrapping needed. Override in subclasses with grid wraparound."""
        return lon

    def get_sonde_geometry(self, sonde_ds, ds, buffer_latlon=0.05, buffer_time_minutes=10):
        """Compute lat/lon/time slices bounding the radiosonde track, in the model grid."""
        sonde_lat = sonde_ds["lat"].values
        sonde_lon = sonde_ds["lon"].values
        sonde_alt = sonde_ds["alt"].values
        sonde_time = sonde_ds["time"].values

        valid = (
            (~np.isnan(sonde_lat)) &
            (~np.isnan(sonde_lon)) &
            (~np.isnan(sonde_alt)) &
            (sonde_lat > -90) & (sonde_lat < 90) &
            (sonde_lon > -180) & (sonde_lon < 180)
        )

        sonde_lon_model, sonde_lat_model = self.transform_lonlat(sonde_lon, sonde_lat)
        sonde_lon_model = self.wrap_longitudes(sonde_lon_model, ds)

        # Calculate bounding box
        lat_min, lat_max = np.min(sonde_lat_model[valid]), np.max(sonde_lat_model[valid])
        lon_min, lon_max = np.min(sonde_lon_model[valid]), np.max(sonde_lon_model[valid])
        alt_min, alt_max = np.min(sonde_alt[valid]), np.max(sonde_alt[valid])
        time_min, time_max = np.min(sonde_time[valid]), np.max(sonde_time[valid])

        buffer_time = np.timedelta64(buffer_time_minutes, 'm')

        return {
            # Slices - for NCO cropping
            "lat_slice": slice(lat_min - buffer_latlon, lat_max + buffer_latlon),
            "lon_slice": slice(lon_min - buffer_latlon, lon_max + buffer_latlon),
            "time_slice": slice(time_min - buffer_time, time_max + buffer_time),
            "alt_min": alt_min,
            "alt_max": alt_max,
            # Sonde track, real-world coords
            "sonde_lat": sonde_lat,
            "sonde_lon": sonde_lon,
            "sonde_alt": sonde_alt,
            "sonde_time": sonde_time,
            "valid": valid,
            # Sonde track, reprojected into model coordinate space
            "sonde_lat_model": sonde_lat_model,
            "sonde_lon_model": sonde_lon_model,
        }

    @staticmethod
    def get_index_bounds(ds, dim_name, coord_slice):
        """Find the integer index range of a coordinate slice within a dataset dimension."""
        if dim_name not in ds.coords:
            return None
        coord_vals = ds[dim_name].values
        # Find positions falling inside the slice boundary
        mask = (coord_vals >= coord_slice.start) & (coord_vals <= coord_slice.stop)
        indices = np.where(mask)[0]
        if len(indices) == 0:
            return None
        return int(indices.min()), int(indices.max())

    def get_dim_bounds(self, ds, slices):
        """Map lat/lon/time slices onto the model's actual grid dimension names."""
        dim_bounds = {}
        for axis, dim_names in self.GRID_DIM_MAP.items():
            coord_slice = slices[f"{axis}_slice"]
            for dim_name in dim_names:
                dim_bounds[dim_name] = self.get_index_bounds(ds, dim_name, coord_slice)
        return dim_bounds

    @staticmethod
    def find_straddle_indices(target_vals, source_vals):
        """For each value in target_vals, find index i into source_vals such that
        source_vals[i] <= target_val <= source_vals[i+1].
        e.g. Find the consecutive pair of cv_lats that straddles t_lat to the N and S.
        Falls back to nearest neighbour if target is outside source range.
        Automatically corrects for any clipping-induced index offset.
        np.clip prevents indexing outside the array bounds
        """
        # Check if t falls between adjacent bounds regardless of direction
        # min_bound <= t <= max_bound
        s_min = np.minimum(source_vals[:-1], source_vals[1:])
        s_max = np.maximum(source_vals[:-1], source_vals[1:])
        
        idx = np.array([
            np.where((s_min <= t) & (t <= s_max))[0][0]
            if np.any((s_min <= t) & (t <= s_max))
            else np.argmin(np.abs(source_vals - t))
            for t in target_vals
        ])
    
        # The clipping operation can shift the relationship between source and target
        # indices by a small integer amount. We try offsets of 0, +1, -1, +2, -2
        # and pick whichever results in the most target points being correctly
        # straddled (i.e. source[i] <= target <= source[i+1] holds for both neighbours).
        best_offset, best_score = 0, -1
        for offset in [0, 1, -1, 2, -2]:
            idx_try = np.clip(idx + offset, 0, len(source_vals) - 2)

            # Determine lower (left) and upper (right) bounds regardless of coordinate direction
            s_left = np.minimum(source_vals[idx_try], source_vals[idx_try + 1])
            s_right = np.maximum(source_vals[idx_try], source_vals[idx_try + 1])
            
            left_ok  = s_left <= target_vals        # left neighbour is to the left
            right_ok = s_right >= target_vals       # right neighbour is to the right
            score = np.sum(left_ok & right_ok)      # count points where BOTH hold
            if score > best_score:
                best_score, best_offset = score, offset
    
        return np.clip(idx + best_offset, 0, len(source_vals) - 2)

    @staticmethod
    def destagger_along_axis(field, idx, axis):
        """Average the two staggered-grid neighbours (idx, idx+1) that straddle
        each target-grid point, collapsing that axis onto the target grid."""
        lower = np.take(field, idx,     axis=axis)
        upper = np.take(field, idx + 1, axis=axis)
        return 0.5 * (lower + upper)

    def find_uv_names(self, ds):
        """Locate the u/v wind component variable names in ds.
        Subclasses override for their own naming convention (STASH codes,
        GRIB shortNames, etc.)."""
        raise NotImplementedError

    def destagger_uv(self, ds):
        """Destagger u/v onto the mass/theta grid, if the model needs it.
        Default: u/v are already collocated with the main grid (e.g. AROME,
        IFS output) - nothing to do. C-grid models (MetUM, WRF) override this."""
        return ds

    def ensure_orography(self, model_ds, orography_path):
        """Return an orography Dataset at orography_path, generating it from
        this model's own output first if the file doesn't already exist."""
        if not os.path.exists(orography_path):
            print(f"\nNo orography file found at {orography_path} - generating one from model data...")
            self.generate_orography_file(model_ds, orography_path)
        return xr.open_dataset(orography_path, decode_times=False)

    def generate_orography_file(self, model_ds, orography_path):
        """Write a standalone orography netCDF file, derived from model_ds.
        Subclasses must override - the source variable name and native grid
        naming are model-specific."""
        raise NotImplementedError

    def load_orography(self, orog_ds, ds):
        """Load and align orography onto ds's horizontal grid, if this model
        needs it for height computation. Default: not needed."""
        return None

    def compute_z_3d(self, ds, orog_aligned):
        """Compute 3D physical height on this model's native vertical levels.
        Subclasses must override - there's no sensible generic default since
        every model's vertical coordinate is different."""
        raise NotImplementedError

    def get_var_name_map(self, ds):
        """Map this model's native variable names to standard sonde-comparison
        names (e.g. {'STASH_...': 'temp', u_var: 'u', ...}). Subclasses override."""
        raise NotImplementedError

    def rotate_winds_to_true(self, results, lon_rot, lat_rot):
        """Rotate u/v from this model's native grid convention to true
        eastward/northward wind, in-place on results['u']/results['v'].
        Default: model grid is already true north/east - nothing to do."""
        return results

    def check_time_overlap(self, ds, time_slice, sonde_tag):
        """Raise a clear error if time_slice doesn't overlap ds's time range,
        instead of letting clip_method silently produce an empty dataset."""
        time_dim = next((d for d in self.GRID_DIM_MAP["time"] if d in ds.dims), None)
        if time_dim is None:
            return
        t_min, t_max = ds[time_dim].values.min(), ds[time_dim].values.max()
        if time_slice.stop < t_min or time_slice.start > t_max:
            raise ValueError(
                f"[{sonde_tag}] Sonde time window {time_slice.start} to {time_slice.stop} "
                f"does not overlap model time range {t_min} to {t_max} (dim '{time_dim}')."
            )
    
class MetUMRun(ModelRun):
    """Reader for MetUM output files, saved per-variable,
    e.g. 'w_ALPS_1km_ERA5_BAS_MetUM_v1_20s_20250225T1200Z.nc'
    """

    # This class has no init and so it will run ModelRun.__init__ instead.
    
    # Name of model
    MODEL_NAME = "MetUM"

    # Define target chunk sizes for each dimension type
    TARGET_CHUNKS = {
        'grid_longitude_t': 200, 'grid_latitude_t': 200,
        'grid_longitude_cu': 200, 'grid_latitude_cu': 200,
        'grid_longitude_cv': 200, 'grid_latitude_cv': 200,
        'eta_theta': 10, 'eta_rho': 10,
        'TS': 1, 'HR': 1
    }

    # Define model grid dimensions
    GRID_DIM_MAP = {
        "lon": ["grid_longitude_t", "grid_longitude_cu", "grid_longitude_cv"],
        "lat": ["grid_latitude_t", "grid_latitude_cu", "grid_latitude_cv"],
        "time": ["TS1", "T1HR"],
    }
    
    def discover_files(self):
        """This function discovers relevant files using glob"""
        # First, create the search string without the variable at the front
        prefix = self.config["prefix"]
        date_time = datetime.datetime.strptime(self.config["date_time"], "%Y%m%dT%H%MZ")        
        all_files = glob.glob(os.path.join(self.directory, f"*_{prefix}_*.nc"))

        possible_files = {}
        for path in all_files:
            basename = os.path.basename(path)
            # Extract datetime string from filename
            parts = basename.removesuffix(".nc").split(f"_{prefix}_")
            if len(parts) != 2:
                continue
            try:
                file_dt = datetime.datetime.strptime(parts[1], "%Y%m%dT%H%MZ")
            except ValueError:
                continue

            if file_dt <= date_time:
                possible_files.setdefault(file_dt, {})[parts[0]] = path
        
        if not possible_files:
            print("No valid model files found")
            return {}
        
        return possible_files[max(possible_files)]

    def _xropen_single(self, path):
        """Helper function to open a single dataset using xarray and run _safely_chunk_var
           (This is essentially a slightly customised xr.open_dataset function)"""
        ds = xr.open_dataset(path)
        for var_name in list(ds.data_vars):
            ds[var_name] = self._safely_chunk_var(ds[var_name])
        return ds

    def _safely_chunk_var(self, var):
        """Helper function to safely chunk a single variable based on its actual dimensions"""
        var_chunks = {}
        for dim in var.dims:
            for key, size in self.TARGET_CHUNKS.items():
                if key in dim:
                    var_chunks[dim] = size
                    break
        return var.chunk(var_chunks) if var_chunks else var
    
    def find_main_variable(self, ds):
        """This function explores the variables present and uses MO STASH codes
        to find the main variable in the file.
        N.B. This assumes only one main variable is present."""
        for v in ds.data_vars:
            long_name = ds[v].attrs.get("long_name", "").lower()
            std_name = ds[v].attrs.get("standard_name", "").lower()
            # CODE HERE NEEDS EDITING TO EXPAND TO TEMPERATURE, U, V etc.
            if ("m01s00i150" in v.lower()
                    or "w compnt" in long_name
                    or std_name == "upward_air_velocity"):
                return v
        # Fallback: Return the first (and sometimes only) data variable in the file
        return list(ds.data_vars)[0]
    
    def find_coord_names(self, ds):
        """Scan the files coordinates to exactract lat, lon, alt"""
        lat_name = "latitude_t" if "latitude_t" in ds.coords else next(
            (c for c in ds.coords if "lat" in c.lower()), None)
        lon_name = "longitude_t" if "longitude_t" in ds.coords else next(
            (c for c in ds.coords if "lon" in c.lower()), None)
        z_name = next((v for v in ds.variables if "zsea" in v.lower()), None)
        return lat_name, lon_name, z_name

    def find_pole_coords(self, ds):
        """Extract rotated pole lat/lon if present, otherwise keep the unrotated default."""
        grid_mapping_var = next(
            (name for name in ['rotated_pole', 'rotated_latitude_longitude'] if name in ds),
            None
        )
        if grid_mapping_var is None:
            return self.pole_lat, self.pole_lon  # no rotation info -> keep default (90, 0)
    
        self.pole_lat = ds[grid_mapping_var].attrs['grid_north_pole_latitude']
        self.pole_lon = ds[grid_mapping_var].attrs['grid_north_pole_longitude']
        return self.pole_lat, self.pole_lon

    def wrap_longitudes(self, lon, ds):
        """Wrap rotated longitudes into the model's native 0-360 grid range."""
        lon_min = float(ds["grid_longitude_t"].min())
        lon_max = float(ds["grid_longitude_t"].max())
        for _ in range(2):
            lon = np.where(lon < lon_min, lon + 360, lon)
            lon = np.where(lon > lon_max, lon - 360, lon)
        return lon

    @staticmethod
    def find_dim(ds, key):
        """Find the actual dim name in ds containing 'key' as a substring"""
        if key in ds.dims:
            return key
        return next((d for d in ds.dims if key in d), None)

    def find_uv_names(self, ds):
        u_var = next(v for v in ds.data_vars if "m01s00i002" in v)
        v_var = next(v for v in ds.data_vars if "m01s00i003" in v)
        return u_var, v_var

    def destagger_uv(self, ds):
        """Destagger u/v from their cu/cv Arakawa-C grids onto the t/theta grid."""
        u_var, v_var = self.find_uv_names(ds)

        # Extract specific level/time dimension names
        theta_dim = self.find_dim(ds, "eta_theta")
        rho_dim   = self.find_dim(ds, "eta_rho")
        time_dim  = self.find_dim(ds, "TS1") or self.find_dim(ds, "T1HR")

        # Load target (mass/theta) and staggered (u/v/rho) coordinate values as numpy arrays
        t_lons     = ds["grid_longitude_t"].values
        t_lats     = ds["grid_latitude_t"].values
        cu_lons    = ds["grid_longitude_cu"].values
        cv_lats    = ds["grid_latitude_cv"].values
        theta_levs = ds[theta_dim].values
        rho_levs   = ds[rho_dim].values

        # Map each target gridpoint to its surrounding staggered grid indices
        u_lon_idx = self.find_straddle_indices(t_lons, cu_lons)
        v_lat_idx = self.find_straddle_indices(t_lats, cv_lats)
        lev_idx   = self.find_straddle_indices(theta_levs, rho_levs)

        # Look up dimension and axis indices for u and v
        u_dims = list(ds[u_var].dims)
        v_dims = list(ds[v_var].dims)
    
        u_lon_axis = u_dims.index("grid_longitude_cu")
        u_lev_axis = u_dims.index(rho_dim)
    
        v_lat_axis = v_dims.index("grid_latitude_cv")
        v_lev_axis = v_dims.index(rho_dim)

        # Define output metadata (target mass-grid coordinates and dimension order)
        t_coords = {
            time_dim: ds[time_dim],
            theta_dim: ds[theta_dim],
            "grid_latitude_t": ds["grid_latitude_t"],
            "grid_longitude_t": ds["grid_longitude_t"],
        }
        out_dims = [time_dim, theta_dim, "grid_latitude_t", "grid_longitude_t"]

        # Interpolate U-wind across longitude then vertical levels, re-wrap into DataArray
        u_h   = self.destagger_along_axis(ds[u_var].values, u_lon_idx, axis=u_lon_axis)
        u_out = self.destagger_along_axis(u_h, lev_idx, axis=u_lev_axis)
        ds[u_var] = xr.DataArray(u_out, dims=out_dims, coords=t_coords)

        # Interpolate V-wind across latitude then vertical levels, re-wrap into DataArray
        v_h   = self.destagger_along_axis(ds[v_var].values, v_lat_idx, axis=v_lat_axis)
        v_out = self.destagger_along_axis(v_h, lev_idx, axis=v_lev_axis)
        ds[v_var] = xr.DataArray(v_out, dims=out_dims, coords=t_coords)

        # Clean up dataset by dropping now-obsolete staggered dimensions
        return ds.drop_dims([
            "grid_latitude_cu", "grid_longitude_cu",
            "grid_latitude_cv", "grid_longitude_cv",
            rho_dim,
        ])

    def generate_orography_file(self, model_ds, orography_path):
        """Build a standalone orography netCDF file from this model run's own
        output, matching the layout of a manually-supplied orography file:
        dims (t, surface, rlat, rlon), variable 'ht', plus a rotated_pole
        grid-mapping variable.
        """
        orog_var = next(v for v in model_ds.data_vars if "m01s00i033" in v)
        orog_da = model_ds[orog_var]

        # Orography is a static 2D field - MetUM output may still carry it with
        # a size-1 time and/or level dim attached, so drop anything that isn't
        # the horizontal grid.
        for dim in list(orog_da.dims):
            if dim not in ("grid_latitude_t", "grid_longitude_t"):
                orog_da = orog_da.isel({dim: 0}, drop=True)

        # Rename back to the rlon/rlat convention used by supplied
        # orography files, so load_orography's existing rename step still works
        orog_da = orog_da.rename({
            "grid_longitude_t": "rlon",
            "grid_latitude_t": "rlat",
        })

        # Re-add singleton surface/t dims so the file has the same dimensionality
        # as a real supplied orography file (load_orography squeezes these off)
        orog_da = orog_da.expand_dims({"surface": [1.0], "t": [0.0]})
        orog_da = orog_da.transpose("t", "surface", "rlat", "rlon")
        orog_da.name = "ht"
        orog_da.attrs.update({
            "long_name": "OROGRAPHY (/STRAT LOWER BC)",
            "standard_name": "surface_altitude",
            "units": "m",
            "grid_mapping": "rotated_pole",
        })

        out_ds = orog_da.to_dataset()
        out_ds["rlon"].attrs.update({
            "long_name": "longitude in rotated pole grid",
            "standard_name": "grid_longitude", "units": "degrees",
        })
        out_ds["rlat"].attrs.update({
            "long_name": "latitude in rotated pole grid",
            "standard_name": "grid_latitude", "units": "degrees",
        })
        out_ds["t"].attrs.update({
            "units": "days since 0000-01-01 00:00:00", "calendar": "none",
        })

        # Rotated-pole grid-mapping variable, using THIS run's own pole coords
        # (self.pole_lat/pole_lon, populated by find_pole_coords in get_dataset)
        out_ds["rotated_pole"] = xr.DataArray(np.array("", dtype="S1"))
        out_ds["rotated_pole"].attrs.update({
            "grid_mapping_name": "rotated_latitude_longitude",
            "grid_north_pole_longitude": self.pole_lon,
            "grid_north_pole_latitude": self.pole_lat,
        })

        out_ds.attrs["history"] = (
            f"Generated from model output ({self.config.get('run_tag', 'unknown run')}) "
            f"by create_synthetic_sonde, {datetime.datetime.now(datetime.timezone.utc).isoformat()}"
        )

        print(f"Writing generated orography file to {orography_path}...")
        out_ds.to_netcdf(orography_path)
    
    def load_orography(self, orog_ds, ds):
        orog_clean = orog_ds["ht"].squeeze(dim=["t", "surface"], drop=True)
        orog_clean = orog_clean.rename({"rlon": "grid_longitude_t", "rlat": "grid_latitude_t"})
        return orog_clean.interp_like(ds)
    
    def compute_z_3d(self, ds, orog_aligned):
        """Computes 3D height for MetUM theta levels without hardcoding grid dimensions."""
        zsea    = [v for v in ds.variables if "zsea" in v.lower() and "theta" in v.lower()
                   and "bound" not in v.lower() and "bnds" not in v.lower()][0]
        
        c_theta = [v for v in ds.variables if "c" in v.lower() and "theta" in v.lower()
                   and "zsea" not in v.lower() and "bound" not in v.lower() and "bnds" not in v.lower()][0]
        
        return ds[zsea] + (ds[c_theta] * orog_aligned)

    def get_var_name_map(self, ds):
        u_var, v_var = self.find_uv_names(ds)
        return {
            "STASH_m01s00i150": "vel_z",
            "STASH_m01s16i004": "temp",
            "STASH_m01s00i004": "PotTemp",
            u_var:              "u",
            v_var:              "v",
            "STASH_m01s00i408": "pressure",
            "STASH_m01s00i010": "specific_humidity",
        }

    @staticmethod
    def _wind_grid_to_true(u_grid, v_grid, lon_rot, lat_rot, pole_lon, pole_lat):
        """
        Rotate wind components from rotated-grid (x_wind/y_wind) to true
        zonal/meridional (eastward_wind/northward_wind).
        lon_rot, lat_rot : sonde position in ROTATED grid coordinates (degrees)
        pole_lon, pole_lat : rotated pole location (degrees)
        """
        lon_r = np.deg2rad(lon_rot)
        lat_r = np.deg2rad(lat_rot)
        pole_lon_r = np.deg2rad(pole_lon)
        pole_lat_r = np.deg2rad(pole_lat)
        sin_c = (np.cos(pole_lat_r) * np.sin(lon_r))
        cos_c = (np.cos(lat_r) * np.sin(pole_lat_r) -
                 np.sin(lat_r) * np.cos(pole_lat_r) * np.cos(lon_r))
        angle = np.arctan2(sin_c, cos_c)
        u_true = u_grid * np.cos(angle) - v_grid * np.sin(angle)
        v_true = u_grid * np.sin(angle) + v_grid * np.cos(angle)
        return u_true, v_true

    def rotate_winds_to_true(self, results, lon_rot, lat_rot):
        results['u'], results['v'] = self._wind_grid_to_true(
            results['u'], results['v'],
            lon_rot, lat_rot,
            self.pole_lon, self.pole_lat
        )
        return results
    
class AROMERun(ModelRun):
    """Reader for AROME output files, saved per-time-step,
    e.g. 'grid.arome-forecast.teamxa001+0012:00.grib'
    """

    # This class has no init and so it will run ModelRun.__init__ instead.

    # Name of model
    MODEL_NAME = "AROME"

    def _xropen_single(self, path):
        """Helper function to open a single dataset using cfgrib
           (This is essentially a slightly customised cfgrib.open_datasets function)"""
        sub_ds = cfgrib.open_datasets(
            path,
            backend_kwargs={"filter_by_keys":
                            {"stepType": "instant",
                             "typeOfLevel": "isobaricInhPa"}},
            chunks={"time": 1, "isobaricInhPa": "auto", "latitude": "auto", "longitude": "auto"}
        )
        return self._merge_by_var(sub_ds)

    def _combine_datasets(self, datasets):
        """For AROME, concatenate along time, since files are per time-step"""
        return xr.concat(datasets, dim="time")
    
    def discover_files(self):
        prefix = self.config["prefix"]
        print(prefix)
        all_files = glob.glob(os.path.join(self.directory, f"{prefix}*.grib"))
        print(all_files)

        possible_files = {}
        for path in all_files:
            key = os.path.basename(path)[-10:-5]
            print("key:", key)
            possible_files[key] = path

        return possible_files

class NEWMODELRun(ModelRun):
    """Reader for NEWMODEL output files, saved per-variable,
    e.g. 'XYZ.nc'
    """

    # This class has no init and so it will run ModelRun.__init__ instead.

    def discover_files(self):
        return
    
    if not ModelRun:
        raise ValueError(f"Error: Incomplete model class selected.")

# Registered available model readers
MODEL_READERS = {
    "metum": MetUMRun,
    "arome": AROMERun,
    # "wrf": WRFRun,  # <-- need to add a WRFRun(ModelRun) subclass
}

station_lookup = {
        "16014": {
            "name": "Sterzing",
            "lat": 46.884,
            "lon": 11.441
        },
        "002KOL": {
            "name": "Kolsass",
            "lat": 47.305,
            "lon": 11.622
        },
        "BOZEN": {
            "name": "Bozen",
            "lat": 46.459,
            "lon": 11.321
        },
    }

metadata = {
    "alt": {
        "long_name": "Geometric height above geoid (WGS84)",
        "units": "m",
        "standard_name": "altitude",
        "axis": "Z",
        "name": "Altitude",
    },
    "timePeriod": {
        "long_name": "Elapsed Time",
        "units": "s",
        "name": "Original Time Dimension from BUFR Data Source"    ,    
    },
    "temp": {
        "long_name": "Air Temperature",
        "units": "K",
        "standard_name": "air_temperature",
        "name": "Temperature",
    },
    "dewp": {
        "long_name": "Dew Point Temperature",
        "units": "K",
        "standard_name": "dew_point_temperature",
        "name": "Dewpoint",
    },
    "lat": {
        "long_name": "Latitude",
        "units": "degrees_north",
        "standard_name": "latitude",
        "axis": "Y",
        "name": "Latitude",
    },
    "lon": {
        "long_name": "Longitude",
        "units": "degrees_east",
        "standard_name": "longitude",
        "axis": "X",
        "name": "Longitude",
    },
    "latitudeDisplacement": {
        "units": "degrees_north",
        "name": "Original Latitude Displacement from BUFR Data Source",
    },
    "longitudeDisplacement": {
        "units": "degrees_east",
        "name": "Original Longitude Displacement from BUFR Data Source",
    },
    "windDir": {
        "long_name": "Wind From Direction",
        "units": "degree",
        "name": "Wind Direction (opposite of heading)",
        "description": "Direction the wind is coming from, measured in degrees clockwise from north. Opposite of the radiosonde heading.",
    },
    "vel_h": {
        "long_name": "Horizontal Wind Velocity",
        "units": "m s-1",
        "standard_name": "wind_speed",
        "name": "Wind Speed",
    },
    "vel_z": {
        "long_name": "Balloon Ascent Rate",
        "units": "m s-1",
        "name": "Upward Balloon Velocity",
    },
    "batt": {
        "units": "volts",
        "name": "Battery Voltage",
    },
    "frame": {
        "units": "counts",
        "name": "Telemetry Data Packet Number",
    },
    "frequency": {
        "units": "MHz",
        "name": "Radio Transmission Frequency",
    },
    "heading": {
        "units": "degrees clockwise from north",
        "name": "Radiosonde Heading",
    },
    "rssi": {
        "units": "dBm",
        "name": "Received Signal Strength Indicator",
    },
    "sats": {
        "units": "counts",
        "name": "Number of GPS Satellites used by Radiosonde",
    },
    "upload_time_delta": {
        "units": "s",
        "name": "Network Upload Latency",
    },
    "uploader_alt": {
        "units": "m",
        "name": "Altitude of Ground Station that Uploaded this Data Packet",
    },
    "pressure": {
        "long_name": "Air Pressure",
        "units": "hPa",
        "standard_name": "air_pressure",
        "name": "Pressure",
    },
    "burst_timer": {
        "units": "s",
        "name": "Time until Automatic Radiosonde TurnOff",
    },
    "snr": {
        "units": "dB",
        "name": "Signal-to-Noise Ratio",
    },
    "tx_frequency": {
        "units": "MHz",
        "name": "Measured Actual Radio Transmission Frequency",
    },
    "relative_humidity": {
        "long_name": "Relative Humidity",
        "units": "%",
        "standard_name": "relative_humidity",
        "name": "Relative Humidity",
    },
    "datetime": {
        "name": "Date and Time",
    },
}

fieldentryorder = {
        "vars": [
            "timePeriod",
            "lat",
            "lon",
            "alt",
            "pressure",
            "temp",
            "dewp",
            "relative_humidity",
            "specific_humidity",
            "vel_h",
            "windDir",
            "u",
            "v",
            "vel_z",
            "vel_z_smoothed",
            "vel_z_prime",
            "vel_z_prime_smoothed",
            "PotTemp",
            "CompRng",
            "CompAz",
            "heading",
            "datetime",
            "qc_flag"
        ],
    
        "globattrs": [
            "Conventions",
            "title",
            "project",
            "source",
            "history",
            "processing_level",
            "acknowledgement",
            "contact",
            "institution",
            "date_created",
            "featureType",
            "time_coverage_start",
            "time_coverage_end",
            "platform",
            "platform_type",
            "platform_altitude",
            "deployment_mode",
            "site_name",
            "location_keywords",
            "geospatial_bounds",
            "sampling_interval",
            "averaging_interval",
            "instrument_manufacturer",
            "instrument_model",
            "instrument_software",
            "instrument_software_version",
            "serial",
            "startdate",
            "starttime",
            "originalFile"
        ],
    }

def create_synthetic_sonde(filepath, config, args):
    """Read a radiosonde netCDF, parse its time, find matching model data,
    and generate the synthetic track."""
    print(f"\nAnalysing Sonde: {os.path.basename(filepath)}...")
    
    # Open the radiosonde NetCDF file
    with xr.open_dataset(filepath) as sonde_ds:
        # Find launch time
        if 'time' in sonde_ds:
            sonde_launch_time = np.datetime64(sonde_ds['time'].values[0])
        
        # Just need to check the formatting of the below...
        #elif sonde_ds.attrs.get('launch_time'):
            #sonde_launch_time = sonde_ds.attrs.get('launch_time')
        #elif sonde_ds.attrs.get('startdate') and sonde_ds.attrs.get('starttime'):
            #sonde_launch_time = sonde_ds.attrs.get('startdate') + sonde_ds.attrs.get('starttime')
        
        else:
            raise KeyError(f"Could not find a valid time variable in {filepath}")

        # Store sonde launch time in same format as model output
        ModelTime = pd.to_datetime(sonde_launch_time,
                                   utc=True).strftime("%Y%m%dT%H%M") + 'Z'
        config["date_time"] = ModelTime
        print("Time in model format:", ModelTime)

        # Store sonde tag for later filenaming
        sonde_tag_pre = os.path.basename(filepath).replace(".nc", "")
        sonde_tag = re.sub(r'_?v\d+.*$', '', sonde_tag_pre)
        print("Sonde tag:", sonde_tag)

        # Use the model specified in the config dict to initialise the right reader class
        reader_class = MODEL_READERS[config["model"]]
        # Feed config into the correct reader_class and assign to the variable 'run'
        run = reader_class(config)

        # Obtain model ds
        model_ds = run.get_dataset()

        # Obtain orography
        if "orography" not in config or not config["orography"]:
            # Extract resolution tag from prefix (e.g. "0p3km" from "w_0p3km_...")
            prefix_parts = config.get("prefix", "").split("_")
            resolution_tag = prefix_parts[1] if len(prefix_parts) > 1 else "unknownres"
            run_tag = config.get("run_tag", config["model"])
            
            config["orography"] = f"orography_{run_tag}_{resolution_tag}.nc"
            print(f"No orography path configured - defaulting to {config['orography']}")
        # Old orography file load system...
        # orog_ds = xr.open_dataset(config["orography"], decode_times=False)
        orog_ds = run.ensure_orography(model_ds, config["orography"])

        # Get sonde geometry in model world, including slices for NCO or XR.SEL
        track = run.get_sonde_geometry(sonde_ds, model_ds)

        # Double check that this sonde was launched inside the simulation timeframe
        run.check_time_overlap(model_ds, track["time_slice"], sonde_tag)

        # Either run NCO to clip files according to sonde's dimension bounds
        if config['clip_method'] == "nco":
            files_dict = run.discover_files()
            dim_bounds = run.get_dim_bounds(model_ds, track)

            # Close model_ds
            model_ds.close()

            # Run ncks
            tmp_output_files = nco_utils.crop_files_with_ncks(files_dict, dim_bounds, sonde_tag, run_tag=config.get("run_tag"))

            print("\nLoading the following files into memory...\n  - " + "\n  - ".join(tmp_output_files))
            with ProgressBar():
                model_ds_clipped = xr.open_mfdataset(tmp_output_files, compat="override").load()

        # Or run XR.SEL to clip files according to sonde's dimension bounds
        elif config['clip_method'] == "xr":
            tmp_path = xr_utils.xr_clip(
                model_ds=model_ds,
                run=run,
                sonde_tag=sonde_tag,
                run_tag=config.get("run_tag"),
                lat_slice=track['lat_slice'],
                lon_slice=track['lon_slice'],
                time_slice=track['time_slice']
            )

            # Close model_ds
            model_ds.close()
            
            print(f"\nLoading the following file into memory...\n  - {tmp_path}")
            with ProgressBar():
                model_ds_clipped = xr.open_dataset(tmp_path).load()

        # Destagger grids if needed, depending on model grids
        model_ds_clipped = run.destagger_uv(model_ds_clipped)

        # Find time and theta dim names
        time_dim  = run.find_dim(model_ds_clipped, "TS1") or run.find_dim(model_ds_clipped, "T1HR")
        theta_dim = run.find_dim(model_ds_clipped, "eta_theta")

        # Build orography and 3D height
        orog_aligned = run.load_orography(orog_ds, model_ds_clipped)
        z_3d = run.compute_z_3d(model_ds_clipped, orog_aligned)

        # Retrieve valid sonde fields
        valid                = track["valid"]
        sonde_times_valid    = sonde_ds["time"].values[valid]
        sonde_times_valid_i8 = sonde_times_valid.astype("datetime64[ns]").astype("int64")
        sonde_alt_valid      = track["sonde_alt"][valid]
        sonde_lat_rot_valid  = track["sonde_lat_model"][valid]
        sonde_lon_rot_valid  = track["sonde_lon_model"][valid]
        
        # Store model fields
        model_times    = model_ds_clipped[time_dim].values
        model_times_i8 = model_times.astype("datetime64[ns]").astype("int64")

        # Old nearest neighbour time interpolation method, now deprecated
        #nearest_t_idx = np.argmin(np.abs(model_times_i8[:, None] - sonde_times_valid_i8[None, :]), axis=0)

        # Find the two model timesteps that bracket each sonde point in time
        upper_bracket_idx = np.searchsorted(model_times_i8, sonde_times_valid_i8)

        # Clip so every point has a valid pair of neighbours to interpolate between,
        # even sonde points that fall before the first or after the last model time
        # (those get pinned to the first/last available bracket instead of erroring).
        upper_bracket_idx = np.clip(upper_bracket_idx, 1, len(model_times) - 1)
        lower_bracket_idx = upper_bracket_idx - 1

        # For each sonde point, work out how far between its two bracketing
        # timesteps it falls, as a fraction from 0.0 (exactly at the lower
        # timestep) to 1.0 (exactly at the upper timestep). This is the
        # linear interpolation weight used to blend the two timesteps.
        lower_time_i8 = model_times_i8[lower_bracket_idx].astype(float)
        upper_time_i8 = model_times_i8[upper_bracket_idx].astype(float)
        # Avoid divide-by-zero in the case that only one model timestep exists
        bracket_duration = np.where(upper_time_i8 != lower_time_i8, upper_time_i8 - lower_time_i8, 1.0)
        weight_toward_upper = np.clip(
            (sonde_times_valid_i8.astype(float) - lower_time_i8) / bracket_duration,
            0.0, 1.0
        )

        # Map model variable names to sonde variable names
        var_name_map = run.get_var_name_map(model_ds_clipped)
        results = {sonde_name: np.full(len(sonde_times_valid), np.nan)
                   for sonde_name in var_name_map.values()}

        print("\nBeginning for loop")
        # Process sonde points in batches grouped by which pair of model timesteps
        # brackets them - e.g. all sonde points falling between model timestep 3
        # and model timestep 4 are handled together, so we only need to run the
        # horizontal interpolation once per bracket, not once per point.
        distinct_lower_indices = np.unique(lower_bracket_idx)

        for lower_idx in distinct_lower_indices:
            upper_idx = lower_idx + 1

            # Which sonde points belong to this particular bracket?
            points_in_this_bracket = lower_bracket_idx == lower_idx
            print(f"Bracket [{lower_idx}, {upper_idx}] / {len(model_times)-1}: "
                  f"{points_in_this_bracket.sum()} sonde points")

            # Pull out this bracket's sonde positions and interpolation weights
            bracket_sonde_lats    = sonde_lat_rot_valid[points_in_this_bracket]
            bracket_sonde_lons    = sonde_lon_rot_valid[points_in_this_bracket]
            bracket_sonde_alts    = sonde_alt_valid[points_in_this_bracket]
            bracket_weight_upper  = weight_toward_upper[points_in_this_bracket]

            lat_da = xr.DataArray(bracket_sonde_lats, dims="sonde_point")
            lon_da = xr.DataArray(bracket_sonde_lons, dims="sonde_point")

            # Height (z_3d) is static - built from orography, not from any model
            # timestep - so it only needs horizontal interpolation, no time interpolation.
            z_at_sonde = z_3d.interp(
                grid_latitude_t=lat_da,
                grid_longitude_t=lon_da,
                method="linear"
            ).compute()

            for var_name, sonde_name in var_name_map.items():
                if var_name not in model_ds_clipped:
                    print(f"WARNING: '{sonde_name}' not found in sonde_ds - skipping model-level interpolation for this variable")
                    continue

                # Horizontally interpolate this variable at BOTH bracketing
                # timesteps separately, at each sonde point's horizontal position.
                var_at_lower_time = model_ds_clipped[var_name].isel({time_dim: lower_idx}).interp(
                    grid_latitude_t=lat_da, grid_longitude_t=lon_da, method="linear"
                ).compute()
                var_at_upper_time = model_ds_clipped[var_name].isel({time_dim: upper_idx}).interp(
                    grid_latitude_t=lat_da, grid_longitude_t=lon_da, method="linear"
                ).compute()

                # Linearly blend the two timesteps using each point's own weight,
                # BEFORE doing the vertical interpolation below.
                var_at_sonde = (
                    (1 - bracket_weight_upper) * var_at_lower_time
                    + bracket_weight_upper * var_at_upper_time
                )

                # Now do the usual per-point vertical (altitude) interpolation,
                # exactly as before - this part is unaffected by the time change.
                interp_vals = np.zeros(len(bracket_sonde_alts))
                for i in range(len(bracket_sonde_alts)):
                    z_col = z_at_sonde.isel(sonde_point=i).values
                    var_col = var_at_sonde.isel(sonde_point=i).values
                    target_alt = bracket_sonde_alts[i]
                    # np.interp requires ascending coordinates - flip if descending
                    if z_col[1] < z_col[0]:
                        z_col = z_col[::-1]
                        var_col = var_col[::-1]
                    interp_vals[i] = np.interp(target_alt, z_col, var_col, left=np.nan, right=np.nan)

                results[sonde_name][points_in_this_bracket] = interp_vals

        # Model-level-sampled sonde: take the sonde's own real observations,
        # degrade them onto the model's vertical resolution at each point, then
        # re-interpolate back onto the sonde's native 1Hz altitude grid. This
        # shows what the *real* atmosphere would look like if only sampled as
        # coarsely as the model resolves it - not a model comparison at all,
        # i.e. just the sonde's own data, vertically smoothed to match model levels.
        results_modellevel = {}
        
        # Set sort order code...
        sort_order = np.argsort(sonde_alt_valid)
        real_alt_sorted = sonde_alt_valid[sort_order]
        
        # Recompute z_at_sonde_all for all sonde points
        # (not a small bracket as previously done for the model comparison)
        lat_da_all = xr.DataArray(sonde_lat_rot_valid, dims="sonde_point")
        lon_da_all = xr.DataArray(sonde_lon_rot_valid, dims="sonde_point")
        z_at_sonde_all = z_3d.interp(
            grid_latitude_t=lat_da_all, grid_longitude_t=lon_da_all, method="linear"
        ).compute()

        for var_name, sonde_name in var_name_map.items():
            if sonde_name not in sonde_ds:
                print(f"WARNING: '{sonde_name}' not found in sonde_ds - skipping model-level coarsening for this variable")
                continue

            # Real, full-resolution sonde profile (must be ascending for np.interp)
            real_var_sorted = sonde_ds[sonde_name].values[valid][sort_order]
            smoothed_vals = np.full(len(sonde_alt_valid), np.nan)
            
            for i in range(len(sonde_alt_valid)):
                z_col = z_at_sonde_all.isel(sonde_point=i).values
                if z_col[1] < z_col[0]:
                    z_col = z_col[::-1]

                # Degrade: sample the REAL sonde profile at model-resolution heights
                var_at_model_levels = np.interp(
                    z_col, real_alt_sorted, real_var_sorted, left=np.nan, right=np.nan
                )
                # Re-interpolate back onto this point's own native altitude
                target_alt = sonde_alt_valid[i]
                smoothed_vals[i] = np.interp(
                    target_alt, z_col, var_at_model_levels, left=np.nan, right=np.nan
                )

            results_modellevel[sonde_name] = smoothed_vals

        results_modellevel['windDirection'] = np.degrees(np.arctan2(-results_modellevel['u'], -results_modellevel['v'])
        ) % 360
        results_modellevel['heading'] = (results_modellevel['windDirection'] + 180) % 360
        results_modellevel['vel_h'] = np.sqrt(results_modellevel['u']**2 + results_modellevel['v']**2)

        # Build and save the model-level-sampled sonde
        if str(config.get("save_modellevel_sonde", False)).lower() in ("1", "true", "yes"):

            # Build results_ds from the model-level-sampled arrays
            results_ds_ml = xr.Dataset(
                {name: (("time",), arr) for name, arr in results_modellevel.items()},
                coords={"time": sonde_times_valid}
            )
            results_ds_ml["time"].attrs.update({
                "standard_name": "time",
                "axis": "T",
                "long_name": "Time"
            })

            # Reindex onto the full original sonde time axis
            results_ds_ml_full = results_ds_ml.reindex(time=sonde_ds['time'].values)

            # Carry over attributes
            for name in results_ds_ml_full.data_vars:
                if name in sonde_ds:
                    results_ds_ml_full[name].attrs.update(sonde_ds[name].attrs)

            # Start from a full copy of the original sonde dataset
            modellevel_ds = sonde_ds.copy(deep=True)

            # Overwrite/add the model-level-sampled variables
            for var in results_ds_ml_full.data_vars:
                modellevel_ds[var] = results_ds_ml_full[var]

            # Update global attrs to make clear this is the sonde's own data,
            # degraded onto the model's vertical resolution - NOT a model comparison
            modellevel_ds.attrs["title"] = sonde_ds.attrs.get("title", "") + " (Model-Level-Sampled Sonde)"
            modellevel_ds.attrs["history"] = (
                modellevel_ds.attrs.get("history", "") +
                f"; Model-level-sampled profile generated {datetime.datetime.now(datetime.timezone.utc).isoformat()}"
            )
            modellevel_ds.attrs["source"] = "Radiosonde observations, vertically degraded to model resolution"

            modellevel_ds = modellevel_ds.rename({"time": "datetime"})
            for var in ["windDir", "relative_humidity", "dewp"]:
                if var in modellevel_ds:
                    modellevel_ds = modellevel_ds.drop_vars(var)
            new_ds_ml = buildxrds(modellevel_ds, args, skip_recomputation=True)

            # Setting filename strings
            timestamp_clean_ml = pd.Timestamp(new_ds_ml.attrs['launch_time']).strftime("%Y%m%d%H%M")
            serial_str_ml = new_ds_ml.attrs.get("serial", "UNKNOWN").strip()
            site_name_str_ml = new_ds_ml.attrs.get("site_name", "UNKNOWN").strip().replace(" ", "").upper()
            site_name_str_ml = site_name_str_ml.ljust(8, "_")

            # Set institution
            if site_name_str_ml == "STERZING":
                data_institution_ml = "NCAS"
                new_ds_ml.attrs["acknowledgement"] = "Original Level 2 data provided by NCAS, UK"
            elif site_name_str_ml == "KOLSASS":
                data_institution_ml = "UIBK"
                new_ds_ml.attrs["acknowledgement"] = "Original Level 1 data provided by UIBK, AT"
            elif site_name_str_ml == "BOZEN":
                data_institution_ml = "KIT_"
                new_ds_ml.attrs["acknowledgement"] = "Original Level 1 data provided by KIT, DE"
            else:
                data_institution_ml = "UNKN"

            # Re-order variables and attributes sensibly
            new_ds_ml = new_ds_ml[[v for v in fieldentryorder["vars"] if v in new_ds_ml]]
            old_attrs_ml = new_ds_ml.attrs
            new_attrs_ml = {k: old_attrs_ml[k] for k in fieldentryorder["globattrs"] if k in old_attrs_ml}
            for k in old_attrs_ml:
                if k not in new_attrs_ml:
                    new_attrs_ml[k] = old_attrs_ml[k]
            new_ds_ml.attrs = new_attrs_ml

            # Clean global attributes that are none-type
            new_ds_ml.attrs = {k: ("" if v is None else v) for k, v in new_ds_ml.attrs.items()}

            # Clean variable-level attributes that are none-type
            for var in new_ds_ml.variables:
                new_ds_ml[var].attrs = {k: ("" if v is None else v) for k, v in new_ds_ml[var].attrs.items()}

            # Pull the resolution tag (e.g. "0p3km") out of the prefix
            prefix_parts_ml = config["prefix"].split("_")
            resolution_tag_ml = prefix_parts_ml[1] if len(prefix_parts_ml) > 1 else "unknownres"
            run_tag_ml = config.get("run_tag", "unknownrun")

            output_filename_ml = (
                f"{timestamp_clean_ml}_{serial_str_ml}_{site_name_str_ml}_{data_institution_ml}_"
                f"{run_tag_ml}_{resolution_tag_ml}_ModelLevel_v0.4.Met.nc"
            )
            print(output_filename_ml)

            new_ds_ml.to_netcdf(output_filename_ml)
            print("Successfully saved netcdf for model-level-sampled sonde")

        # Rotate u/v to true eastward/northward wind if needed
        results = run.rotate_winds_to_true(results, sonde_lon_rot_valid, sonde_lat_rot_valid)

        # Calculate Wind Direction from true u and v
        results['windDirection'] = np.degrees(np.arctan2(-results['u'], -results['v'])) % 360
        # Calculate heading from windDirection
        results['heading'] = (results['windDirection'] + 180) % 360
        # Calculate the horizontal velocity from true u and v
        results['vel_h'] = np.sqrt(results['u']**2 + results['v']**2)

        # Build results_ds from interpolated arrays in the results dict
        results_ds = xr.Dataset(
            {name: (("time",), arr) for name, arr in results.items()},
            coords={"time": sonde_times_valid}
        )
        results_ds["time"].attrs.update({
            "standard_name": "time",
            "axis": "T",
            "long_name": "Time"
        })
        
        # Reindex onto the full original sonde time axis
        results_ds_full = results_ds.reindex(time=sonde_ds['time'].values)
        
        # Carry over attributes
        for name in results_ds_full.data_vars:
            if name in sonde_ds:
                results_ds_full[name].attrs.update(sonde_ds[name].attrs)
                
        # Start from a full copy of the original sonde dataset
        model_equivalent_ds = sonde_ds.copy(deep=True)
        
        # Overwrite/add the model-interpolated variables
        for var in results_ds_full.data_vars:
            model_equivalent_ds[var] = results_ds_full[var]
            
        # Update global attrs to make clear this is model data, not observations
        model_equivalent_ds.attrs["title"] = sonde_ds.attrs.get("title", "") + f" ({run.MODEL_NAME}-Interpolated Equivalent)"
        model_equivalent_ds.attrs["history"] = (
            model_equivalent_ds.attrs.get("history", "") +
            f"; {run.MODEL_NAME} Model-equivalent profile generated {datetime.datetime.now(datetime.timezone.utc).isoformat()}"
        )
        model_equivalent_ds.attrs["source"] = f"{run.MODEL_NAME} Model Output"

        model_equivalent_ds = model_equivalent_ds.rename({"time": "datetime"})
        for var in ["windDir", "relative_humidity", "dewp"]:
            if var in model_equivalent_ds:
                model_equivalent_ds = model_equivalent_ds.drop_vars(var)
        new_ds = buildxrds(model_equivalent_ds, args, skip_recomputation=True)

        # Setting filename strings
        timestamp_clean = pd.Timestamp(new_ds.attrs['launch_time']).strftime("%Y%m%d%H%M")
        serial_str = new_ds.attrs.get("serial", "UNKNOWN").strip()
        site_name_str = new_ds.attrs.get("site_name", "UNKNOWN").strip().replace(" ", "").upper()
        site_name_str = site_name_str.ljust(8, "_") # Pad to 8 characters for consistent filename formatting
        
        # Set institution
        if site_name_str == "STERZING":
            data_institution = "NCAS"
            new_ds.attrs["acknowledgement"] = "Original Level 2 data provided by NCAS, UK"
        elif site_name_str == "KOLSASS":
            data_institution = "UIBK"
            new_ds.attrs["acknowledgement"] = "Original Level 1 data provided by UIBK, AT"
        elif site_name_str == "BOZEN":
            data_institution = "KIT_"
            new_ds.attrs["acknowledgement"] = "Original Level 1 data provided by KIT, DE"
        else: data_institution = "UNKN"

        # Re-order variables and attributes sensibly
        new_ds = new_ds[[v for v in fieldentryorder["vars"] if v in new_ds]]
        old_attrs = new_ds.attrs
        new_attrs = {k: old_attrs[k] for k in fieldentryorder["globattrs"] if k in old_attrs}
        
        # Add any remaining attributes at the end
        for k in old_attrs:
            if k not in new_attrs:
                new_attrs[k] = old_attrs[k]
        new_ds.attrs = new_attrs
        
        # Clean global attributes that are none-type
        new_ds.attrs = {k: ("" if v is None else v) for k, v in new_ds.attrs.items()}
        
        # Clean variable-level attributes that are none-type
        for var in new_ds.variables:
            new_ds[var].attrs = {k: ("" if v is None else v) for k, v in new_ds[var].attrs.items()}

        # Pull the resolution tag (e.g. "0p3km") out of the prefix
        prefix_parts = config["prefix"].split("_")
        resolution_tag = prefix_parts[1] if len(prefix_parts) > 1 else "unknownres"
        run_tag = config.get("run_tag", "unknownrun")
        
        output_filename = (
            f"{timestamp_clean}_{serial_str}_{site_name_str}_{data_institution}_"
            f"{run_tag}_{resolution_tag}_v0.4.Met.nc"
        )
        print(output_filename)

        new_ds.to_netcdf(output_filename)
        print("Successfully saved netcdf for simulated sonde")

        fig, ax = plt.subplots(figsize=(6, 8))
        ax.plot(sonde_ds['u'], sonde_ds['alt'], color='crimson', label='U obs')
        ax.plot(new_ds['u'], sonde_ds['alt'], color='crimson', linestyle=':', label='U model')
        ax.plot(sonde_ds['v'], sonde_ds['alt'], color='dodgerblue', label='V obs')
        ax.plot(new_ds['v'], sonde_ds['alt'], color='dodgerblue', linestyle=':', label='V model')
        ax.set_xlabel('Wind (m/s)')
        ax.set_ylabel('Altitude (m)')
        ax.legend()
        plt.tight_layout()
        plt.savefig(f"quickuvcomparison_{sonde_tag}_{config.get('run_tag', 'unknownrun')}.png", dpi=600)

# ============= #
# MAIN FUNCTION #
# ============= #

def main(args):
    # Keep track of created files
    created_files = []

    # Set config file if it exists
    if args.config_file:
        config = read_config_file(args.config_file)
    elif CONFIG:
        config = CONFIG
    else:
        config = None
    
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

                        # RUN CODE HERE! #
                        create_synthetic_sonde(filepath, config, args)
                        created_files.append(filepath)

            else:
                # Processing radiosonde data from file
                print(f"Processing file: {f}")

                # RUN CODE HERE! #
                create_synthetic_sonde(f, config, args)
                created_files.append(f)

    # Success message depending on how many were processed
    if len(created_files) == 1:
        print("Synthetic radiosonde netCDF file created successfully.")
    elif len(created_files) > 1:
        print("Synthetic radiosonde netCDF files created successfully.")
    else:
        print("No files created.") # Should not normally happen
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Radiosonde Through Model Observational Equivalents"
    )

    group = parser.add_mutually_exclusive_group(required=True)

    group.add_argument(
        '-f', '--files',
        nargs='+',
        help='One or more local radiosonde file paths or directories'
    )

    parser.add_argument(
        '-c', '--config-file',
        help='Optional config file to override the one hardcoded into the script'
    )

    parser.add_argument(
        '-l', '--launch-site',
        help='Optional launch site name to force into the created netCDF file'
    )

    parser.add_argument(
        '-d', '--dewpoint-method',
        default='vaisala',
        type=lambda s: s.lower(),
        choices=['vaisala', 'default', 'sa90', 'aerk'],
        help='Specify which method should be used to compute the dew point in its absence (default: vaisala)'
    )

    args = parser.parse_args()

    main(args)