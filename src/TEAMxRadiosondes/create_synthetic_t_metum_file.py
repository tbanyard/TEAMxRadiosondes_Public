import datetime
import numpy as np
import netCDF4 as nc

# 1. Geometry Setup (40x40 spatial layout matching 't' and 'theta' structures)
num_lats = 40
num_lons = 40
num_levels = 60  # Matches TH_1_60_eta_theta
num_times = 4    # 4 timestamps

grid_lat = np.linspace(3.0, 3.4, num_lats)
grid_lon = np.linspace(-11.2, -10.8, num_lons)

lon_center, lat_center = 11.43, 46.89
lon_t, lat_t = np.meshgrid(
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
filename = "t_ALPS_1km_ERA5_BAS_MetUM_v1_20s_20250226T1200Z.nc"
rootgrp = nc.Dataset(filename, "w", format="NETCDF4")

# Set Dimensions matching your ncdump exactly
rootgrp.createDimension("grid_longitude_t", num_lons)
rootgrp.createDimension("grid_latitude_t", num_lats)
rootgrp.createDimension("bounds2", 2)
rootgrp.createDimension("TH_1_60_eta_theta", num_levels)
rootgrp.createDimension("TS1", None)

# 3. Grid Mapping Metadata Variable
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

# 2D Unrotated Spatial Fields
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

# 5. Create Potential Temperature (Theta) Variable (STASH_m01s00i004)
theta_var = rootgrp.createVariable(
    "STASH_m01s00i004", 
    "f4", 
    ("TS1", "TH_1_60_eta_theta", "grid_latitude_t", "grid_longitude_t"),
    fill_value=-1073741824.0
)
theta_var.long_name = "THETA AFTER TIMESTEP"
theta_var.standard_name = "air_potential_temperature"
theta_var.units = "K"
theta_var.coordinates = "longitude_t latitude_t TH_1_60_zsea_theta TH_1_60_C_theta TH_1_60_model_level_number"
theta_var.cell_methods = "TS1: point"
theta_var.grid_mapping = "rotated_latitude_longitude"
theta_var.um_version = "13.5"
theta_var.um_stash_source = "m01s00i004"
theta_var.packing_method = "none"

# 6. Create Absolute Temperature Variable (STASH_m01s16i004)
t_var = rootgrp.createVariable(
    "STASH_m01s16i004", 
    "f4", 
    ("TS1", "TH_1_60_eta_theta", "grid_latitude_t", "grid_longitude_t"),
    fill_value=-1073741824.0
)
t_var.long_name = "TEMPERATURE ON THETA LEVELS"
t_var.standard_name = "air_temperature"
t_var.units = "K"
t_var.coordinates = "longitude_t latitude_t TH_1_60_zsea_theta TH_1_60_C_theta TH_1_60_model_level_number"
t_var.cell_methods = "TS1: point"
t_var.grid_mapping = "rotated_latitude_longitude"
t_var.um_version = "13.5"
t_var.um_stash_source = "m01s16i004"
t_var.packing_method = "none"

# 7. Generate Balanced Thermodynamic Arrays (T profiles drive derived Potential Temp)
np.random.seed(303)
synthetic_theta = np.zeros((num_times, num_levels, num_lats, num_lons), dtype="f4")
synthetic_t = np.zeros((num_times, num_levels, num_lats, num_lons), dtype="f4")

# Environmental baseline reference constants
t_surface = 288.15     # K (15 degrees C)
tropopause_alt = 11000.0  # m
lapse_rate = 0.0065     # K/m (ICAO Standard Atmosphere)

for t in range(num_times):
    for z in range(num_levels):
        height = zsea_vals[z]
        
        # Determine ambient temperature based on atmospheric boundary layers
        if height <= tropopause_alt:
            base_t = t_surface - (lapse_rate * height)
        else:
            base_t = t_surface - (lapse_rate * tropopause_alt)  # Isothermal lower stratosphere
        
        # Reconstruct standard pressure scaling variable to find a valid potential temperature relation
        p_ratio = np.exp(-height / 7400.0)
        base_theta = base_t * (1.0 / p_ratio) ** (287.0 / 1004.0)
        
        # Apply standard small spatial variations and random noise
        noise_t = np.random.normal(0, 0.1, size=(num_lats, num_lons))
        noise_theta = noise_t * (1.0 / p_ratio) ** (287.0 / 1004.0)
        
        synthetic_t[t, z, :, :] = base_t + noise_t
        synthetic_theta[t, z, :, :] = base_theta + noise_theta

t_var[:] = synthetic_t
theta_var[:] = synthetic_theta

# Global Attributes
rootgrp.Conventions = "CF-1.6"
rootgrp.source = "Met Office Unified Model v13.5"
rootgrp.history = f"Synthetic file generated on {datetime.datetime.now().strftime('%Y-%m-%d')}"

rootgrp.close()
print(f"Successfully generated dual-variable mockup temperature file: {filename}")