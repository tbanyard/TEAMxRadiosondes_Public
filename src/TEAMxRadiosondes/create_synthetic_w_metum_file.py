import datetime
import numpy as np
import netCDF4 as nc

# 1. Geometry Setup (Sterzing 20x20km Grid at 1km high-res)
num_lats = 40
num_lons = 40
num_levels = 60  # Matches TH_1_60
num_times = 4   # 4 timestamps (e.g., 6-hourly intervals)

# Create a small grid of rotated coordinates (centered around mock values)
grid_lat = np.linspace(3.0, 3.4, num_lats)
grid_lon = np.linspace(-11.2, -10.8, num_lons)

# Meshgrid for true geographical coordinates (Sterzing ~46.89 N, 11.43 E)
lon_center, lat_center = 11.43, 46.89
lon_t, lat_t = np.meshgrid(
    np.linspace(lon_center - 0.15, lon_center + 0.15, num_lons),
    np.linspace(lat_center - 0.1, lat_center + 0.1, num_lats)
)

# Vertical coordinates (Logarithmic hybrid levels up to ~40km)
eta_vals = np.linspace(0.0, 1.0, num_levels)
zsea_vals = np.logspace(1, 4.6, num_levels)  # 10m up to ~40,000m
c_vals = np.exp(-zsea_vals / 5000.0)         # Orographic decay factor
model_levels = np.arange(1, num_levels + 1)

# Time axis (Seconds since baseline to match TS1 syntax)
time_vals = np.array([0.0, 21600.0, 43200.0, 64800.0]) # 0h, 6h, 12h, 18h

# 2. Initialize NetCDF File
filename = "w_ALPS_1km_ERA5_BAS_MetUM_v1_20s_20250226T1200Z.nc"
rootgrp = nc.Dataset(filename, "w", format="NETCDF4")

# Set Dimensions matching ncdump
rootgrp.createDimension("grid_longitude_t", num_lons)
rootgrp.createDimension("grid_latitude_t", num_lats)
rootgrp.createDimension("bounds2", 2)
rootgrp.createDimension("TH_1_60_eta_theta", num_levels)
rootgrp.createDimension("TS1", None)

# 3. Create Grid Mapping Metadata Variable
rot_pole = rootgrp.createVariable("rotated_latitude_longitude", "c")
rot_pole.grid_mapping_name = "rotated_latitude_longitude"
rot_pole.grid_north_pole_latitude = 43.64305005
rot_pole.grid_north_pole_longitude = 190.8167457

# 4. Create Coordinate Variables & Bounds
glon = rootgrp.createVariable("grid_longitude_t", "f8", ("grid_longitude_t",))
glon.standard_name = "grid_longitude"
glon.long_name = "longitude in rotated pole grid"
glon.units = "degrees"
glon.axis = "X"
glon.bounds = "grid_longitude_t_bounds"
glon[:] = grid_lon

glon_bnd = rootgrp.createVariable("grid_longitude_t_bounds", "f8", ("grid_longitude_t", "bounds2"))
# Simple mock cell boundaries
glon_bnd[:, 0] = grid_lon - 0.005
glon_bnd[:, 1] = grid_lon + 0.005

glat = rootgrp.createVariable("grid_latitude_t", "f8", ("grid_latitude_t",))
glat.standard_name = "grid_latitude"
glat.long_name = "latitude in rotated pole grid"
glat.units = "degrees"
glat.axis = "Y"
glat.bounds = "grid_latitude_t_bounds"
glat[:] = grid_lat

glat_bnd = rootgrp.createVariable("grid_latitude_t_bounds", "f8", ("grid_latitude_t", "bounds2"))
glat_bnd[:, 0] = grid_lat - 0.005
glat_bnd[:, 1] = grid_lat + 0.005

# 2D True Lat/Lon Arrays
true_lon = rootgrp.createVariable("longitude_t", "f8", ("grid_latitude_t", "grid_longitude_t"))
true_lon.standard_name = "longitude"
true_lon.long_name = "longitude"
true_lon.units = "degrees_east"
true_lon[:] = lon_t

true_lat = rootgrp.createVariable("latitude_t", "f8", ("grid_latitude_t", "grid_longitude_t"))
true_lat.standard_name = "latitude"
true_lat.long_name = "latitude"
true_lat.units = "degrees_north"
true_lat[:] = lat_t

# Vertical Metadata Elements
eta = rootgrp.createVariable("TH_1_60_eta_theta", "f8", ("TH_1_60_eta_theta",))
eta.standard_name = "atmosphere_hybrid_height_coordinate"
eta.long_name = "eta value of theta levels"
eta.axis = "Z"
eta.positive = "up"
eta.bounds = "TH_1_60_eta_theta_bounds"
eta[:] = eta_vals

eta_bnd = rootgrp.createVariable("TH_1_60_eta_theta_bounds", "f8", ("TH_1_60_eta_theta", "bounds2"))

zsea = rootgrp.createVariable("TH_1_60_zsea_theta", "f8", ("TH_1_60_eta_theta",))
zsea.standard_name = "height_above_reference_ellipsoid"
zsea.long_name = "Height above mean sea level"
zsea.units = "m"
zsea.positive = "up"
zsea.bounds = "TH_1_60_zsea_theta_bounds"
zsea[:] = zsea_vals

zsea_bnd = rootgrp.createVariable("TH_1_60_zsea_theta_bounds", "f8", ("TH_1_60_eta_theta", "bounds2"))

c_frac = rootgrp.createVariable("TH_1_60_C_theta", "f8", ("TH_1_60_eta_theta",))
c_frac.long_name = "Fraction of orographic height"
c_frac.units = "1"
c_frac.bounds = "TH_1_60_C_theta_bounds"
c_frac[:] = c_vals

c_bnd = rootgrp.createVariable("TH_1_60_C_theta_bounds", "f8", ("TH_1_60_eta_theta", "bounds2"))

lvl_num = rootgrp.createVariable("TH_1_60_model_level_number", "i4", ("TH_1_60_eta_theta",))
lvl_num.standard_name = "model_level_number"
lvl_num.long_name = "model theta levels (Charney-Phillips grid)"
lvl_num.units = "1"
lvl_num.positive = "up"
lvl_num[:] = model_levels

# Time Coordinate
time_var = rootgrp.createVariable("TS1", "f8", ("TS1",))
time_var.standard_name = "time"
time_var.units = "seconds since 2025-02-26 12:00:00"
time_var.axis = "T"
time_var.calendar = "gregorian"
time_var[:] = time_vals

# 5. Create the W-Wind Main STASH Variable
w_var = rootgrp.createVariable(
    "STASH_m01s00i150", 
    "f4", 
    ("TS1", "TH_1_60_eta_theta", "grid_latitude_t", "grid_longitude_t"),
    fill_value=-1073741824.0
)
w_var.long_name = "W COMPNT OF WIND AFTER TIMESTEP"
w_var.standard_name = "upward_air_velocity"
w_var.units = "m s-1"
w_var.coordinates = "longitude_t latitude_t TH_1_60_zsea_theta TH_1_60_C_theta TH_1_60_model_level_number"
w_var.cell_methods = "TS1: point"
w_var.grid_mapping = "rotated_latitude_longitude"
w_var.um_version = "13.5"
w_var.um_stash_source = "m01s00i150"
w_var.packing_method = "none"

# 6. Generate Mathematical Updraft Wave Data
np.random.seed(42)
synthetic_w = np.zeros((num_times, num_levels, num_lats, num_lons), dtype="f4")

for t in range(num_times):
    phase = t * 0.4
    for z in range(num_levels):
        # Emulate lee waves over Alpine terrain that decay with high altitude
        wave = np.sin(lon_t * 60 + phase) * np.cos(lat_t * 60) * (zsea_vals[z] / 1200.0) * np.exp(-zsea_vals[z] / 6000.0)
        noise = np.random.normal(0, 0.04, size=(num_lats, num_lons))
        synthetic_w[t, z, :, :] = wave + noise

w_var[:] = synthetic_w

# Global Attributes
rootgrp.Conventions = "CF-1.6"
rootgrp.source = "Met Office Unified Model v13.5"
rootgrp.history = f"Synthetic file generated on {datetime.datetime.now().strftime('%Y-%m-%d')}"

rootgrp.close()
print(f"Successfully generated clean mockup file: {filename}")