import datetime
import numpy as np
import netCDF4 as nc

# 1. Geometry Setup (Using the same 40x40 spatial bounds but mapped to 'cu' and 'rho' syntax)
num_lats = 40
num_lons = 40
num_levels = 60  # Matches TH_1_60_eta_rho
num_times = 4    # 4 test timestamps

grid_lat = np.linspace(3.0, 3.4, num_lats)
grid_lon = np.linspace(-11.2, -10.8, num_lons)

lon_center, lat_center = 11.43, 46.89
lon_cu, lat_cu = np.meshgrid(
    np.linspace(lon_center - 0.15, lon_center + 0.15, num_lons),
    np.linspace(lat_center - 0.1, lat_center + 0.1, num_lats)
)

# Rho levels use slightly different heights or staggering in a true model, 
# but we mock them using a clean profile up to ~40km
eta_vals = np.linspace(0.0, 1.0, num_levels)
zsea_vals = np.logspace(1, 4.6, num_levels)
c_vals = np.exp(-zsea_vals / 5000.0)
model_levels = np.arange(1, num_levels + 1)

time_vals = np.array([0.0, 21600.0, 43200.0, 64800.0])

# 2. Initialize NetCDF File
filename = "u_ALPS_1km_ERA5_BAS_MetUM_v1_20s_20250226T1200Z.nc"
rootgrp = nc.Dataset(filename, "w", format="NETCDF4")

# Set Dimensions matching your text specification
rootgrp.createDimension("grid_longitude_cu", num_lons)
rootgrp.createDimension("grid_latitude_cu", num_lats)
rootgrp.createDimension("bounds2", 2)
rootgrp.createDimension("TH_1_60_eta_rho", num_levels)
rootgrp.createDimension("TS1", None)

# 3. Grid Mapping Metadata Variable
rot_pole = rootgrp.createVariable("rotated_latitude_longitude", "c")
rot_pole.grid_mapping_name = "rotated_latitude_longitude"
rot_pole.grid_north_pole_latitude = 43.64305005
rot_pole.grid_north_pole_longitude = 190.8167457

# 4. Create Coordinate Variables & Bounds (Updated to *_cu naming)
glon = rootgrp.createVariable("grid_longitude_cu", "f8", ("grid_longitude_cu",))
glon.standard_name = "grid_longitude"
glon.long_name = "longitude in rotated pole grid"
glon.units = "degrees"
glon.axis = "X"
glon.bounds = "grid_longitude_cu_bounds"
glon[:] = grid_lon

glon_bnd = rootgrp.createVariable("grid_longitude_cu_bounds", "f8", ("grid_longitude_cu", "bounds2"))
glon_bnd[:, 0] = grid_lon - 0.005
glon_bnd[:, 1] = grid_lon + 0.005

glat = rootgrp.createVariable("grid_latitude_cu", "f8", ("grid_latitude_cu",))
glat.standard_name = "grid_latitude"
glat.long_name = "latitude in rotated pole grid"
glat.units = "degrees"
glat.axis = "Y"
glat.bounds = "grid_latitude_cu_bounds"
glat[:] = grid_lat

glat_bnd = rootgrp.createVariable("grid_latitude_cu_bounds", "f8", ("grid_latitude_cu", "bounds2"))
glat_bnd[:, 0] = grid_lat - 0.005
glat_bnd[:, 1] = grid_lat + 0.005

# 2D Unrotated Coordinate Arrays
true_lon = rootgrp.createVariable("longitude_cu", "f8", ("grid_latitude_cu", "grid_longitude_cu"))
true_lon.standard_name = "longitude"
true_lon.long_name = "longitude"
true_lon.units = "degrees_east"
true_lon[:] = lon_cu

true_lat = rootgrp.createVariable("latitude_cu", "f8", ("grid_latitude_cu", "grid_longitude_cu"))
true_lat.standard_name = "latitude"
true_lat.long_name = "latitude"
true_lat.units = "degrees_north"
true_lat[:] = lat_cu

# Vertical Metadata Elements (Updated to *_rho naming)
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

# 5. Create the U-Wind Main STASH Variable (STASH_m01s00i002)
u_var = rootgrp.createVariable(
    "STASH_m01s00i002", 
    "f4", 
    ("TS1", "TH_1_60_eta_rho", "grid_latitude_cu", "grid_longitude_cu"),
    fill_value=-1073741824.0
)
u_var.long_name = "U COMPNT OF WIND AFTER TIMESTEP"
u_var.standard_name = "x_wind"
u_var.units = "m s-1"
u_var.coordinates = "longitude_cu latitude_cu TH_1_60_zsea_rho TH_1_60_C_rho TH_1_60_model_level_number"
u_var.cell_methods = "TS1: point"
u_var.grid_mapping = "rotated_latitude_longitude"
u_var.um_version = "13.5"
u_var.um_stash_source = "m01s00i002"
u_var.packing_method = "none"

# 6. Generate Realistic Synoptic Zonal Wind Flow Data (Westerlies jet profile)
np.random.seed(101)
synthetic_u = np.zeros((num_times, num_levels, num_lats, num_lons), dtype="f4")

for t in range(num_times):
    for z in range(num_levels):
        # Build a background westerly jet that peaks near the tropopause (~11-12 km)
        height = zsea_vals[z]
        jet_profile = 15.0 * np.exp(-((height - 11000.0) / 6000.0)**2) + 5.0
        
        # Add slight horizontal shear across coordinates
        spatial_shear = (lon_cu - lon_center) * 4.0
        noise = np.random.normal(0, 0.15, size=(num_lats, num_lons))
        
        synthetic_u[t, z, :, :] = jet_profile + spatial_shear + noise

u_var[:] = synthetic_u

# Global Attributes
rootgrp.Conventions = "CF-1.6"
rootgrp.source = "Met Office Unified Model v13.5"
rootgrp.history = f"Synthetic file generated on {datetime.datetime.now().strftime('%Y-%m-%d')}"

rootgrp.close()
print(f"Successfully generated clean mockup U file: {filename}")