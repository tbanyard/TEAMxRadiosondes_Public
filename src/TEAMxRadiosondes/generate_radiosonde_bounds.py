import os
import numpy as np
import xarray as xr
from datetime import datetime

data_dir = "/home/tim/Documents/Projects/TEAMxRadiosondes/data/"

nph_dir = data_dir + 'Sterzing/NPH_newparsed_netcdfs/'
pr_dir  = data_dir + 'Sterzing/netcdf_processed/'
ks_dir  = data_dir + 'Kolsass/netcdf_processed/'
bz_dir = data_dir + 'Bozen/netcdf_processed/'

NPH_files = [f for f in os.listdir(nph_dir) if f.endswith("_v1.1.TPB.nc")]
PR_files  = [f for f in os.listdir(pr_dir)  if f.endswith("_v2.0.TPB.nc")]
KS_files  = [f for f in os.listdir(ks_dir)  if f.endswith("_v2.0.TPB.nc")]
BZ_files  = [f for f in os.listdir(bz_dir)  if f.endswith("_v2.0.TPB.nc")]

# --- STEP 1: build dictionary with priority to v1.1 ---
def get_serial(fname):
    return fname.split("_")[1]

file_dict = {}

# First add v2.0
for f in PR_files:
    serial = get_serial(f)
    file_dict[serial] = os.path.join(pr_dir, f)

# Then overwrite with v1.1 (priority)
for f in NPH_files:
    serial = get_serial(f)
    file_dict[serial] = os.path.join(nph_dir, f)

for f in KS_files:
    serial = get_serial(f)
    
    if serial in file_dict:
        print(f"Warning: duplicate serial across sites: {serial}")
    
    file_dict[serial] = os.path.join(ks_dir, f)

for f in BZ_files:
    serial = get_serial(f)
    
    if serial in file_dict:
        print(f"Warning: duplicate serial across sites: {serial}")
    
    file_dict[serial] = os.path.join(bz_dir, f)

# --- STEP 2: process files ---
rows = []

for serial, filepath in file_dict.items():
    fname = os.path.basename(filepath)

    # YYYYmmDDHHMM from filename
    launch_str = fname.split("_")[0]
    launch_dt = datetime.strptime(launch_str, "%Y%m%d%H%M")

    ds = xr.open_dataset(filepath)

    lat = ds["lat"].values
    lon = ds["lon"].values
    alt = ds["alt"].values
    time = ds["time"].values  # numpy datetime64

    # valid mask (avoid NaNs)
    # valid = (~np.isnan(lat)) & (~np.isnan(lon)) & (~np.isnan(alt))
    valid = (
        (~np.isnan(lat)) &
        (~np.isnan(lon)) &
        (~np.isnan(alt)) &
        (lat > -90) & (lat < 90) &
        (lon > -180) & (lon < 180)
    )

    if not np.any(valid):
        continue

    idx = np.where(valid)[0]

    first = idx[0]
    last  = idx[-1]

    # start/end
    start_time = np.datetime_as_string(time[first], unit='s').replace("-", "").replace(":", "").replace("T", "")
    end_time   = np.datetime_as_string(time[last],  unit='s').replace("-", "").replace(":", "").replace("T", "")

    start_lat, start_lon, start_alt = lat[first], lon[first], alt[first]
    end_lat,   end_lon,   end_alt   = lat[last],  lon[last],  alt[last]

    # bounding box
    lat_min = np.min(lat[valid])
    lat_max = np.max(lat[valid])
    lon_min = np.min(lon[valid])
    lon_max = np.max(lon[valid])

    site = ds.attrs.get("site_name", "UNKNOWN")

    rows.append((
        launch_dt,
        f"{launch_str} {serial} {site} "
        f"{start_time} {start_lat:.3f} {start_lon:.3f} {start_alt:.1f} "
        f"{end_time} {end_lat:.3f} {end_lon:.3f} {end_alt:.1f} "
        f"{lat_min:.3f} {lat_max:.3f} {lon_min:.3f} {lon_max:.3f}"
    ))

    ds.close()

# --- STEP 3: sort chronologically ---
rows.sort(key=lambda x: x[0])

# --- STEP 4: write file ---
output_file = "radiosonde_summary.txt"

with open(output_file, "w") as f:
    f.write("YYYYmmDDHHMM SERIAL SITE START LAT LON ALT END LAT LON ALT LAT_MIN LAT_MAX LON_MIN LON_MAX\n")
    for _, line in rows:
        f.write(line + "\n")

output_file = "radiosonde_summary.txt"

lat_min_all = []
lat_max_all = []
lon_min_all = []
lon_max_all = []
alt_start_all = []
alt_end_all = []

with open(output_file, "r") as f:
    next(f)  # skip header
    for line in f:
        parts = line.split()

        lat_min_all.append(float(parts[11]))
        lat_max_all.append(float(parts[12]))
        lon_min_all.append(float(parts[13]))
        lon_max_all.append(float(parts[14]))

        alt_start_all.append(float(parts[6]))
        alt_end_all.append(float(parts[10]))

LAT_MIN = min(lat_min_all)
LAT_MAX = max(lat_max_all)
LON_MIN = min(lon_min_all)
LON_MAX = max(lon_max_all)
ALT_MIN = min(alt_start_all)
ALT_MAX = max(alt_end_all)

print(LAT_MIN, LAT_MAX, LON_MIN, LON_MAX, ALT_MIN, ALT_MAX)

kml_2d = f"""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
<Document>
    <name>Radiosonde Bounding Box</name>

    <Placemark>
        <name>2D Bounding Box</name>
        <Style>
            <LineStyle>
                <color>ff0000ff</color>
                <width>3</width>
            </LineStyle>
            <PolyStyle>
                <color>330000ff</color>
            </PolyStyle>
        </Style>

        <Polygon>
            <outerBoundaryIs>
                <LinearRing>
                    <coordinates>
                        {LON_MIN},{LAT_MIN},0
                        {LON_MAX},{LAT_MIN},0
                        {LON_MAX},{LAT_MAX},0
                        {LON_MIN},{LAT_MAX},0
                        {LON_MIN},{LAT_MIN},0
                    </coordinates>
                </LinearRing>
            </outerBoundaryIs>
        </Polygon>
    </Placemark>

</Document>
</kml>
"""

with open("bounding_box_2D.kml", "w") as f:
    f.write(kml_2d)

kml_3d = f"""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
<Document>
    <name>3D Radiosonde Bounding Box</name>

    <Style id="boxStyle">
        <LineStyle>
            <color>ff00ffff</color>
            <width>2</width>
        </LineStyle>
        <PolyStyle>
            <color>3300ffff</color>
        </PolyStyle>
    </Style>

    <!-- Bottom -->
    <Placemark><styleUrl>#boxStyle</styleUrl>
    <Polygon><altitudeMode>absolute</altitudeMode>
    <outerBoundaryIs><LinearRing><coordinates>
        {LON_MIN},{LAT_MIN},{ALT_MIN}
        {LON_MAX},{LAT_MIN},{ALT_MIN}
        {LON_MAX},{LAT_MAX},{ALT_MIN}
        {LON_MIN},{LAT_MAX},{ALT_MIN}
        {LON_MIN},{LAT_MIN},{ALT_MIN}
    </coordinates></LinearRing></outerBoundaryIs>
    </Polygon></Placemark>

    <!-- Top -->
    <Placemark><styleUrl>#boxStyle</styleUrl>
    <Polygon><altitudeMode>absolute</altitudeMode>
    <outerBoundaryIs><LinearRing><coordinates>
        {LON_MIN},{LAT_MIN},{ALT_MAX}
        {LON_MAX},{LAT_MIN},{ALT_MAX}
        {LON_MAX},{LAT_MAX},{ALT_MAX}
        {LON_MIN},{LAT_MAX},{ALT_MAX}
        {LON_MIN},{LAT_MIN},{ALT_MAX}
    </coordinates></LinearRing></outerBoundaryIs>
    </Polygon></Placemark>

    <!-- Walls -->
    <!-- South -->
    <Placemark><styleUrl>#boxStyle</styleUrl>
    <Polygon><altitudeMode>absolute</altitudeMode>
    <outerBoundaryIs><LinearRing><coordinates>
        {LON_MIN},{LAT_MIN},{ALT_MIN}
        {LON_MAX},{LAT_MIN},{ALT_MIN}
        {LON_MAX},{LAT_MIN},{ALT_MAX}
        {LON_MIN},{LAT_MIN},{ALT_MAX}
        {LON_MIN},{LAT_MIN},{ALT_MIN}
    </coordinates></LinearRing></outerBoundaryIs>
    </Polygon></Placemark>

    <!-- North -->
    <Placemark><styleUrl>#boxStyle</styleUrl>
    <Polygon><altitudeMode>absolute</altitudeMode>
    <outerBoundaryIs><LinearRing><coordinates>
        {LON_MIN},{LAT_MAX},{ALT_MIN}
        {LON_MAX},{LAT_MAX},{ALT_MIN}
        {LON_MAX},{LAT_MAX},{ALT_MAX}
        {LON_MIN},{LAT_MAX},{ALT_MAX}
        {LON_MIN},{LAT_MAX},{ALT_MIN}
    </coordinates></LinearRing></outerBoundaryIs>
    </Polygon></Placemark>

    <!-- West -->
    <Placemark><styleUrl>#boxStyle</styleUrl>
    <Polygon><altitudeMode>absolute</altitudeMode>
    <outerBoundaryIs><LinearRing><coordinates>
        {LON_MIN},{LAT_MIN},{ALT_MIN}
        {LON_MIN},{LAT_MAX},{ALT_MIN}
        {LON_MIN},{LAT_MAX},{ALT_MAX}
        {LON_MIN},{LAT_MIN},{ALT_MAX}
        {LON_MIN},{LAT_MIN},{ALT_MIN}
    </coordinates></LinearRing></outerBoundaryIs>
    </Polygon></Placemark>

    <!-- East -->
    <Placemark><styleUrl>#boxStyle</styleUrl>
    <Polygon><altitudeMode>absolute</altitudeMode>
    <outerBoundaryIs><LinearRing><coordinates>
        {LON_MAX},{LAT_MIN},{ALT_MIN}
        {LON_MAX},{LAT_MAX},{ALT_MIN}
        {LON_MAX},{LAT_MAX},{ALT_MAX}
        {LON_MAX},{LAT_MIN},{ALT_MAX}
        {LON_MAX},{LAT_MIN},{ALT_MIN}
    </coordinates></LinearRing></outerBoundaryIs>
    </Polygon></Placemark>

</Document>
</kml>
"""

with open("bounding_box_3D.kml", "w") as f:
    f.write(kml_3d)