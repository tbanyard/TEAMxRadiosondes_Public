import datetime
import numpy as np
import netCDF4 as nc

# 1. Geometry Setup (40x40 spatial layout mapped to 'cv' and 'rho' structures)
num_lats = 40
num_lons = 40
num_levels = 60  # Matches TH_1_60_eta_rho
num_times = 4    # 4 timestamps

grid_lat = np.linspace(3.0, 3.4, num_lats)
grid_lon = np.linspace(-11.2, -10.8, num_lons)

lon_center, lat_center = 11.43, 46.89
lon_cv, lat_cv = np.meshgrid(
    np.linspace(lon_center - 0.15, lon_center + 0.15, num_lons),
    np.linspace(lat_center - 0.1, lat_center + 0.1, num_lats)
)

# Vertical coordinates matching previous setups
eta_vals = np.linspace(0.0, 1.0, num_levels)
zsea_vals = np.logspace(1, 4.6, num_levels)
c_vals = np.exp(-zsea_vals / 5000.0)
model_levels = np.arange(1, num_levels + 1)

time_vals = np.array([0.0, 21600.0, 43200.0, 64800.0])

# 2. Initialize NetCDF File
filename = "v_ALPS_1km_ERA5_BAS_MetUM_v1_20s_20250226T1200Z.nc"
rootgrp = nc.Dataset(filename, "w", format="NETCDF4")

# Set Dimensions matching V structure
rootgrp.createDimension("grid_longitude_cv", num_lons)
rootgrp.createDimension("grid_latitude_cv", num_lats)
rootgrp.createDimension("bounds2", 2)
rootgrp.createDimension("TH_1_60_eta_rho", num_levels)
rootgrp.createDimension("TS1", None)

# 3. Grid Mapping Metadata Variable
rot_pole = rootgrp.createVariable("rotated_latitude_longitude", "c")
rot_pole.grid_mapping_name = "rotated_latitude_longitude"
rot_pole.grid_north_pole_latitude = 43.64305005
rot_pole.grid_north_pole_longitude = 190.8167457

# 4. Create Coordinate Variables & Bounds
glon = rootgrp.createVariable("grid_longitude_cv", "f8", ("grid_longitude_cv",))
glon.standard_name = "grid_longitude"
glon.long_name = "longitude in rotated pole grid"
glon.units = "degrees"
glon.axis = "X"
glon.bounds = "grid_longitude_cv_bounds"
glon[:] = grid_lon

glon_bnd = rootgrp.createVariable("grid_longitude_cv_bounds", "f8", ("grid_longitude_cv", "bounds2"))
glon_bnd[:, 0] = grid_lon - 0.005
glon_bnd[:, 1] = grid_lon + 0.005

glat = rootgrp.createVariable("grid_latitude_cv", "f8", ("grid_latitude_cv",))
glat.standard_name = "grid_latitude"
glat.long_name = "latitude in rotated pole grid"
glat.units = "degrees"
glat.axis = "Y"
glat.bounds = "grid_latitude_cv_bounds"
glat[:] = grid_lat

glat_bnd = rootgrp.createVariable("grid_latitude_cv_bounds", "f8", ("grid_latitude_cv", "bounds2"))
glat_bnd[:, 0] = grid_lat - 0.005
glat_bnd[:, 1] = grid_lat + 0.005

# 2D Unrotated Spatial Fields
true_lon = rootgrp.createVariable("longitude_cv", "f8", ("grid_latitude_cv", "grid_longitude_cv"))
true_lon.standard_name = "longitude"
true_lon.long_name = "longitude"
true_lon.units = "degrees_east"
true_lon[:] = lon_cv

true_lat = rootgrp.createVariable("latitude_cv", "f8", ("grid_latitude_cv", "grid_longitude_cv"))
true_lat.standard_name = "latitude"
true_lat.long_name = "latitude"
true_lat.units = "degrees_north"
true_lat[:] = lat_cv

# Vertical Metadata Elements
eta = rootgrp.createVariable("TH_1_60_eta_rho", "f8", ("TH_1_60_eta_rho",))
eta.standard_name = "atmosphere_hybrid_height_coordinate"
eta.long_name = "eta value of rho levels"
eta.axis = "Z"
eta.positive = "up"
eta.bounds = "TH_1_60_eta_rho_bounds"
eta[:] = eta_vals

eta_bnd = rootgrp.createVariable("TH_1_60_eta_rho_bounds", "f8", ("TH_1_60_eta_rho", "bounds2"))

zsea = rootgrp.createVariable("TH_1_60_zsea_rho", "f8", ("TH_1_60_eta_rho",))
zsea.standard_name = "height_above_reference_ellipsoid"
zsea.long_name = "Height above mean sea level"
zsea.units = "m"
zsea.positive = "up"
zsea.bounds = "TH_1_60_zsea_rho_bounds"
zsea[:] = zsea_vals

zsea_bnd = rootgrp.createVariable("TH_1_60_zsea_rho_bounds", "f8", ("TH_1_60_eta_rho", "bounds2"))

c_frac = rootgrp.createVariable("TH_1_60_C_rho", "f8", ("TH_1_60_eta_rho",))
c_frac.long_name = "Fraction of orographic height"
c_frac.units = "1"
c_frac.bounds = "TH_1_60_C_rho_bounds"
c_frac[:] = c_vals

c_bnd = rootgrp.createVariable("TH_1_60_C_rho_bounds", "f8", ("TH_1_60_eta_rho", "bounds2"))

lvl_num = rootgrp.createVariable("TH_1_60_model_level_number", "i4", ("TH_1_60_eta_rho",))
lvl_num.standard_name = "model_level_number"
lvl_num.long_name = "model rho levels (Charney-Phillips grid)"
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

# 5. Create V-Wind Variable (STASH_m01s00i003)
v_var = rootgrp.createVariable(
    "STASH_m01s00i003", 
    "f4", 
    ("TS1", "TH_1_60_eta_rho", "grid_latitude_cv", "grid_longitude_cv"),
    fill_value=-1073741824.0
)
v_var.long_name = "V COMPNT OF WIND AFTER TIMESTEP"
v_var.standard_name = "y_wind"
v_var.units = "m s-1"
v_var.coordinates = "longitude_cv latitude_cv TH_1_60_zsea_rho TH_1_60_C_rho TH_1_60_model_level_number"
v_var.cell_methods = "TS1: point"
v_var.grid_mapping = "rotated_latitude_longitude"
v_var.um_version = "13.5"
v_var.um_stash_source = "m01s00i003"
v_var.packing_method = "none"

# 6. Generate Realistic Synoptic Meridional Wind Flow Data
np.random.seed(404)
synthetic_v = np.zeros((num_times, num_levels, num_lats, num_lons), dtype="f4")

for t in range(num_times):
    for z in range(num_levels):
        height = zsea_vals[z]
        # Build a weaker meridional flow compared to the westerly u-jet, 
        # but with a clear synoptic wave/shear pattern across latitude.
        base_v = 2.0 * np.sin(lat_cv * 30 + (t * 0.2)) * (height / 8000.0) * np.exp(-height / 15000.0)
        
        noise = np.random.normal(0, 0.1, size=(num_lats, num_lons))
        synthetic_v[t, z, :, :] = base_v + noise

v_var[:] = synthetic_v

# Global Attributes
rootgrp.Conventions = "CF-1.6"
rootgrp.source = "Met Office Unified Model v13.5"
rootgrp.history = f"Synthetic file generated on {datetime.datetime.now().strftime('%Y-%m-%d')}"

rootgrp.close()
print(f"Successfully generated mockup V wind file: {filename}")