import os
import sys
from TEAMxRadiosondes.funcs import fixnanswithmean
import sondehub
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.signal import savgol_filter
import xarray as xr
import pandas as pd
import datetime
import time
import argparse
from pathlib import Path
import pdbufr
import re

# Define directories
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, '..', '..', 'data')

# Define global dictionaries
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

def haversine(lats, lons):
    """Returns the distance between two lat/lon points in kilometers."""
    R = 6371 # Earth's radius in km
    lat1, lat2 = np.radians(lats)
    lon1, lon2 = np.radians(lons)
    # Haversine formula
    a = (np.sin((lat2-lat1)/2) ** 2) + (np.cos(lat1) * \
        np.cos(lat2) * (np.sin((lon2-lon1)/2) ** 2))
    c = 2 * np.arcsin(np.sqrt(a))
    d = R * c
    return d

def find_nearest_station(lat, lon, station_lookup, max_distance_km=20):
    """Return the nearest station name and distance, or None if too far."""
    nearest_station = None
    min_distance = float('inf')

    for code, info in station_lookup.items():
        dist = haversine((lat, info['lat']), (lon, info['lon']))
        if dist < min_distance:
            min_distance = dist
            nearest_station = info['name']

    if min_distance <= max_distance_km:
        return nearest_station, min_distance
    else:
        print(f"Distance to {nearest_station} is {min_distance:.2f} km "
              f"which is above the threshold of {max_distance_km} km. "
              f"Setting site_name to None...")
        return None, min_distance

def set_site_name(ds, force_site_name=None):
    """
    Add a 'site_name' attribute to the dataset.

    Parameters:
        ds : xarray.Dataset
        force_site_name : str, optional
            If provided, use this as the site_name regardless of launch_site.
    """
    current_site_name = ds.attrs.get("site_name")

    if force_site_name is not None:
        ds.attrs["site_name"] = force_site_name
        return ds
    
    if current_site_name and current_site_name != "Unknown":
        return ds

    # Check if 'launch_site' exists (this is the alphanumerical code)
    launch_site = ds.attrs.get("launch_site")
    if launch_site is not None:
        # Lookup name from dictionary, default to 'Unknown' if not found
        ds.attrs["site_name"] = station_lookup.get(str(launch_site), {}).get("name", "Unknown")
    else:
        if 'lat' in ds and 'lon' in ds:
            nearest_site = find_nearest_station(ds['lat'][0].values, ds['lon'][0].values, station_lookup)
            if nearest_site[1] < 10:
                ds.attrs["site_name"] = nearest_site[0]
            elif 10 <= nearest_site[1] < 20:
                print("Warning: Trialling a max distance of 20 km for site name since nearest site found is >10 km away...")
                time.sleep(3)
                nearest_site = find_nearest_station(ds['lat'][0].values, ds['lon'][0].values, station_lookup, max_distance_km=20)
                ds.attrs["site_name"] = nearest_site[0]
            else:
                ds.attrs["site_name"] = "Unknown"
        else:
            ds.attrs["site_name"] = "Unknown"
    
    return ds

def bearing(lat1, lon1, lat2, lon2):
    dlon = np.radians(lon2 - lon1)
    lat1, lat2 = np.radians(lat1), np.radians(lat2)
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1)*np.sin(lat2) - np.sin(lat1)*np.cos(lat2)*np.cos(dlon)
    return (np.degrees(np.arctan2(x, y)) + 360) % 360

def parse_latlon(s):
    s = s.replace("°", "").strip()
    hem = s[-1].upper()
    val = float(s[:-1])
    return -val if hem in ["S", "W"] else val

def parse_temp_c(s):
    s = s.decode() if isinstance(s, (bytes, np.bytes_)) else str(s)
    num = float(re.search(r"[-+]?\d*\.?\d+", s).group())
    return num - 273.15 if "k" in s.lower() else num

def parse_wind_dir(s):
    s = s.decode() if isinstance(s, (bytes, np.bytes_)) else str(s)
    return int(re.search(r"[-+]?\d*\.?\d+", s).group())

def safely_rename_attr(ds, old, new):
    if old in ds.attrs:
        ds.attrs[new] = ds.attrs.pop(old)

def extract_number(s):
    """Return the first number found in a string as float."""
    if isinstance(s, (bytes, np.bytes_)):
        s = s.decode(errors="ignore")
    m = re.search(r"[-+]?\d*\.?\d+", s)
    return float(m.group()) if m else None

def saturation_vapour_pressure_AERK(T_C, p_hPa):
    """
    Calculate saturation vapour pressure using AERK/AERKI formulas.
    
    Parameters:
        T_C: float or ndarray
            Air temperature in °C
        p_hPa: float or ndarray
            Air pressure in hPa
            
    Returns:
        e_hPa: float or ndarray
            Saturation vapor pressure in hPa
    """
    T = np.array(T_C, dtype=float)
    p = np.array(p_hPa, dtype=float)

    e_s = np.where(
        T >= 0,
        6.1094 * np.exp((17.6251 * T) / (243.04 + T)),  # AERK over water
        6.1121 * np.exp((22.587 * T) / (273.86 + T))   # AERKI over ice
    )

    # Enhancement factor for moist air
    enhancement = np.where(
        T >= 0,
        1.00071 * np.exp(0.0000045 * p),
        0.99882 * np.exp(0.000008 * p)
    )

    return e_s * enhancement

def dew_point_from_RH(T_C, RH, p_hPa, method="vaisala"):
    """
    Calculate dew point using a variety of methods.
    
    Parameters:
        T_C: float or ndarray
            Air temperature in °C
        RH: float or ndarray
            Relative humidity in %
        p_hPa: float or ndarray
            Air pressure in hPa
        method: string
            Method to use for dewpoint calculation
            Choose from vaisala, aerk or sa90
            
    Returns:
        Td_C: float or ndarray
            Dew point temperature in °C
        description: string
            Description of the method used
    """

    # Normalise method name
    method = method.lower().strip()
    if method == "default":
        method = "vaisala"

    if method == "aerk":
        description = "Computed using the saturation vapour‐pressure expansion from Alduchov, O. A. and Eskridge, R. E. (1996) (J. Appl. Meteor., 1996, 35 (4), 601–609)"
        e_s = saturation_vapour_pressure_AERK(T_C, p_hPa)
        e = RH / 100.0 * e_s

        # Inverse AERK/AERKI for Td
        Td_C = np.where(
            T_C >= 0,
            243.04 * np.log(e / 6.1094) / (17.6251 - np.log(e / 6.1094)),
            273.86 * np.log(e / 6.1121) / (22.587 - np.log(e / 6.1121))
        )

    elif method == "vaisala":
        description = "Computed using Vaisala Magnus-type approximation (B210973EN-F)"
        A = 6.116441   # hPa
        m = 7.591386   # dimensionless
        Tn = 240.7263  # °C

        e_s = A * 10 ** (m * T_C / (T_C + Tn))
        e = RH / 100.0 * e_s

        # Compute dewpoint temperature (°C)
        Td_C = (Tn * np.log10(e / A)) / (m - np.log10(e / A))

    elif method == "sa90":
        description = "Computed using the Sonntag, D. (1990) ITS-90-consistent vapour-pressure formulation (Z. Meteor., 40, 340–344)."
        a = 17.62
        b = 243.12

        gamma = (a * T_C) / (b + T_C) + np.log(RH / 100.0)
        Td_C = (b * gamma) / (a - gamma)

    else:
        print("Dew point Method Invalid")
        sys.exit(1)

    return Td_C, description

def convertprnctonetcdf(filepath, args):
    ds = xr.open_dataset(filepath)
    
    # Time dimension = launch + elapsed seconds
    launch_dt = ds["time"].values[0]
    elapsed = ds["elapsed_time"].values.flatten().astype(float)
    datetime = pd.to_datetime(launch_dt) + pd.to_timedelta(elapsed, unit='s')

    ds = ds.assign_coords(datetime=("time", datetime))
    ds = ds.swap_dims({"time": "datetime"})
    ds = ds.drop_vars("time")
    ds = ds.set_coords(["datetime"])

    ref_time_str = pd.Timestamp(launch_dt).strftime("%Y-%m-%d %H:%M:%S")

    ds["datetime"].encoding = {
        "units": f"seconds since {ref_time_str}",
        "calendar": "proleptic_gregorian",
        "dtype": "float64",
    }

    # Tidy variable names
    rename_map = {
        "air_pressure": "pressure",
        "air_temperature": "airTemperature",
        "dew_point_temperature": "dewpointTemperature",
        "relative_humidity": "relative_humidity",
        "altitude": "alt",
        "latitude": "lat",
        "longitude": "lon",
        "wind_speed": "vel_h",
        "upward_balloon_velocity": "vel_z",
        "wind_from_direction": "windDirection",
        "elapsed_time": "timePeriod",
        "time": "sourceTime"
    }
    ds = ds.rename({k: v for k, v in rename_map.items() if k in ds})
    ds = ds.drop_vars(["day_of_year", "year", "month", "day", "hour", "minute", "second"], errors="ignore")

    # Modifying DS to fit template:
    ds = buildxrds(ds, args)
    ds["timePeriod"].attrs.update({
        "name": "Elapsed Time Since Launch from Original Data Source"
    })

    # Fixing attributes
    ds.attrs["serial"] = ds.attrs.get("comment")[25:33]
    ds = ds.reset_coords()
    for v in ds.variables:
        for attr in ["cell_methods", "valid_min", "valid_max", "coordinates"]:
            ds[v].attrs.pop(attr, None)

    # List of global attributes to delete
    for attr in ["instrument_serial_number",
                 "creator_name",
                 "creator_email",
                 "creator_url",
                 "processing_software_version",
                 "calibration_sensitivity",
                 "calibration_certification_date",
                 "calibration_certification_url",
                 "project_principal_investigator",
                 "project_principal_investigator_email",
                 "project_principal_investigator_url",
                 "licence",
                 "amf_vocabularies_release",
                 "comment",
                 "product_version",
                 "last_revised_date",
                 "processing_software_url"]:
        ds.attrs.pop(attr, None)

    # Adding administrative and final missing attributes
    ds.attrs["Conventions"] = "CF-1.13"
    ds.attrs["processing_level"] = np.int32(3)
    ds.attrs["originalFile"] = os.path.basename(filepath)
    ds.attrs["geospatial_bounds"] = ds.attrs.pop("geospacial_bounds")
    ds.attrs['launch_time'] = pd.Timestamp(ds.attrs["time_coverage_start"], tz="UTC").isoformat()
    ds.attrs["source"] = "NCAS netCDF file"

    # Setting filename strings and acknowledgements
    fp = Path(filepath)
    stem = fp.stem
    timestamp = stem.split("_")[2] 
    timestamp_clean = timestamp.replace("-", "")
    timestamp_clean = timestamp_clean[:-2] # Remove seconds for netCDF file name
    serial_str = ds.attrs.get("serial", "UNKNOWN").strip()
    site_name_str = ds.attrs.get("site_name", "UNKNOWN").strip().replace(" ", "").upper()
    if site_name_str == "STERZING":
        data_institution = "NCAS"
        ds.attrs["acknowledgement"] = "Original Level 2 data provided by NCAS, UK"
    elif site_name_str == "KOLSASS":
        data_institution = "UIBK"
        ds.attrs["acknowledgement"] = "Original Level 2 data provided by UIBK, AT"
    elif site_name_str == "BOZEN":
        data_institution = "KIT_"
        ds.attrs["acknowledgement"] = "Original Level 2 data provided by KIT, DE"
    site_name_str = site_name_str.ljust(8, "_") # Pad to 8 characters for consistent filename formatting

    # Re-order attributes sensibly
    ds = ds[[v for v in fieldentryorder["vars"] if v in ds]]
    old_attrs = ds.attrs
    new_attrs = {k: old_attrs[k] for k in fieldentryorder["globattrs"] if k in old_attrs}

    # Add any remaining attributes at the end
    for k in old_attrs:
        if k not in new_attrs:
            new_attrs[k] = old_attrs[k]
    ds.attrs = new_attrs

    # Clean global attributes that are none-type
    ds.attrs = {k: ("" if v is None else v) for k, v in ds.attrs.items()}

    # Clean variable-level attributes that are none-type
    for var in ds.variables:
        ds[var].attrs = {k: ("" if v is None else v) for k, v in ds[var].attrs.items()}

    # Saving netCDF file
    #output_filename = fp.parent / f"{timestamp_clean}{serial_str}_NCAS_all.nc"
    output_filename = fp.parent / f"{timestamp_clean}_{serial_str}_{site_name_str}_{data_institution}_v2.0.TPB.nc"
    ds.to_netcdf(output_filename)
    print(f"NetCDF file saved: {output_filename}")

def convertnphnctonetcdf(filepath, args):
    ds = xr.open_dataset(filepath)

    if "ElapsedTime" in ds:
        # Use the dimension that ElapsedTime actually uses
        main_dim = ds["ElapsedTime"].dims[1]
    else:
        # Fallback: pick the longest dimension if ElapsedTime missing
        dims_sorted = sorted(ds.dims.items(), key=lambda x: x[1], reverse=True)
        main_dim = dims_sorted[0][0]

    # Extract launch date/time strings safely
    def _bytes_to_str(arr):
        return "".join([x.decode() if isinstance(x, (bytes, np.bytes_)) else str(x)
                        for x in arr.flatten()])

    date_str = _bytes_to_str(ds["Balloonreleasedate"].values)
    time_str = _bytes_to_str(ds["Balloonreleasetime"].values)

    launch_dt = pd.to_datetime(f"{date_str} {time_str}")

    # Time dimension = launch + elapsed seconds
    elapsed = ds["ElapsedTime"].values.flatten().astype(float)
    datetime = pd.to_datetime(launch_dt) + pd.to_timedelta(elapsed, unit='s')

    # Add datetime coordinate
    ds = ds.assign_coords(datetime=(main_dim, datetime))

    # Move main dimension to 'datetime'
    ds = ds.swap_dims({main_dim: "datetime"})
    ds = ds.drop_vars([main_dim], errors="ignore")

    # Collapse variables to 1D along datetime
    for v in list(ds.data_vars):
        if "datetime" in ds[v].dims:
            ds[v] = (["datetime"], ds[v].values.flatten())

    # Promote text-like variables (|S1 etc.) to global attributes
    def _bytes_to_text(arr):
        """Safely decode a |S1 array into a readable string, keeping degree symbols."""
        chars = []
        for x in arr.flatten():
            if isinstance(x, (bytes, np.bytes_)):
                chars.append(x.decode('latin-1', errors='ignore'))  # latin-1 keeps '°'
            else:
                chars.append(str(x))
        return "".join(chars).strip()

    for v in list(ds.data_vars):
        arr = ds[v]
        if (arr.dtype.kind in ["S", "U"]) and ("datetime" not in arr.dims) and (arr.size < 200):
            try:
                text = _bytes_to_text(arr.values)
                if text:
                    ds.attrs[v] = text
                ds = ds.drop_vars(v)
            except Exception:
                pass

    # Tidy variable names
    rename_map = {
        "P": "pressure",
        "Temp": "temp",
        "Dewp": "dewp",
        "RH": "relative_humidity",
        "HeightMSL": "alt",
        "Lat": "lat",
        "Lon": "lon",
        "Speed": "vel_h",
        "AscRate": "vel_z",
        "Dir": "windDirection",
        "Ecomp": "u",
        "Ncomp": "v",
        "ElapsedTime": "timePeriod",
        "Time": "sourceTime"
    }
    ds = ds.rename({k: v for k, v in rename_map.items() if k in ds})

    # Sort by datetime & final clean
    ds = ds.sortby("datetime")
    ds = ds.squeeze(drop=True)
    ds = ds.set_coords(["datetime"])
    
    # Modifying DS to fit template:
    ds = buildxrds(ds, args)
    ds["sourceTime"].attrs.update({
        "units": "MATLAB serial days",
        "name": "Original MATLAB Time Dimension parsed from Data Source",
        "note": "The MATLAB units are days there have been since 0000-01-01 00:00:00"
    })
    ds["timePeriod"].attrs.update({
        "name": "Elapsed Time Since Launch from Original Data Source"
    })
    ds["GpsHeight"].attrs.update({
        "name": "GpsHeight from Original Data Source",
        "note": "The alt variable is preferred to this one, if it is available, as it is likely to be more accurate"
    })

    # Fixing attributes
    safely_rename_attr(ds, "Balloonreleasetime", "balloonreleasetime_nph")
    safely_rename_attr(ds, "Balloonreleasedate", "balloonreleasedate_nph")
    safely_rename_attr(ds, "CreationDate", "originalFileCreationDate_nph")
    safely_rename_attr(ds, "OriginalFile", "originalFile_nph")
    safely_rename_attr(ds, "Softwareversion", "softwareVersion")
    safely_rename_attr(ds, "Laptop", "laptop")
    ds.attrs.pop("CreatedBy", None)
    ds.attrs.pop("long_title", None)

    ds.attrs["heightOfStationGroundAboveMeanSeaLevel"] = extract_number(ds.attrs.pop("Releasepointheightfromsealevel"))
    ds.attrs["source"] = "NPH netCDF file"
    ds.attrs["project"] = "Multi-scale transport and exchange processes in the atmosphere over mountains – programme and experiment (TEAMx)"
    ds.attrs["featureType"] = "trajectory"
    ds.attrs["startLatitude"] = parse_latlon(ds.attrs["Releasepointlatitude"])
    ds.attrs["startLongitude"] = parse_latlon(ds.attrs["Releasepointlongitude"])
    ds.attrs["surfaceTemperature_C"] = parse_temp_c(ds.attrs["Surfacetemperature"])
    ds.attrs["surfaceWindDir"] = parse_wind_dir(ds.attrs["Surfacewinddirection"])
    ds.attrs.pop("Releasepointlatitude", None)
    ds.attrs.pop("Releasepointlongitude", None)
    ds.attrs.pop("Surfacetemperature", None)
    ds.attrs.pop("Surfacewinddirection", None)
    safely_rename_attr(ds, "Surfacehumidity", "surfaceHumidity")
    safely_rename_attr(ds, "Surfacepressure", "surfacePressure")

    # Sort conflicting serial numbers
    s2 = ds.attrs.get("Sondeserialnumber")
    s1 = ds.attrs.get("Serial")

    # Warn if conflicting serial numbers
    if s1 != s2:
        print(f"\n\033[1mWARNING:\033[0m "
            f"Serial '{s1}' is different from Sondeserialnumber '{s2}'. "
            f"Using '{s2}' as Serial.\n")
        
    # Overwrite Serial and remove Sondeserialnumber
    ds.attrs["Serial"] = s2
    ds.attrs.pop("Sondeserialnumber", None)
    ds.attrs["serial"] = ds.attrs.pop("Serial")

    # Overwrite site_name and remove Site
    if "Site" in ds.attrs:
        ds.attrs["site_name"] = ds.attrs.get("Site")
        ds.attrs.pop("Site", None)

    if "LaunchTime" in ds.attrs:
        # Parse the existing string
        t = pd.to_datetime(ds.attrs["LaunchTime"], format="%d-%b-%Y %H:%M:%S", errors="coerce")
        if pd.notna(t):
            # Convert to ISO 8601 with explicit UTC timezone
            ds.attrs["launch_time"] = t.strftime("%Y-%m-%dT%H:%M:%S+00:00")
            # Remove the old attribute
            del ds.attrs["LaunchTime"]
        else:
            print("Could not parse LaunchTime:", ds.attrs["LaunchTime"])

    # Adding administrative attributes
    ds.attrs["Conventions"] = "CF-1.13"
    ds.attrs["processing_level"] = np.int32(3)
    ds.attrs["originalFile"] = os.path.basename(filepath)
    #ds.attrs["geospatial_bounds"] = #

    # Setting filename strings and acknowledgements
    fp = Path(filepath)
    stem = fp.stem
    timestamp = stem.replace("Sonde_", "")[:-1]
    timestamp_clean = timestamp.replace("-", "")
    serial_str = ds.attrs.get("serial", "UNKNOWN").strip()
    site_name_str = ds.attrs.get("site_name", "UNKNOWN").strip().replace(" ", "").upper()
    if site_name_str == "STERZING":
        data_institution = "NCAS"
        ds.attrs["acknowledgement"] = "Original Level 2 data provided by NCAS, UK"
    elif site_name_str == "KOLSASS":
        data_institution = "UIBK"
        ds.attrs["acknowledgement"] = "Original Level 1 data provided by UIBK, AT"
    elif site_name_str == "BOZEN":
        data_institution = "KIT_"
        ds.attrs["acknowledgement"] = "Original Level 1 data provided by KIT, DE"
    site_name_str = site_name_str.ljust(8, "_") # Pad to 8 characters for consistent filename formatting

    # Re-order variables and attributes sensibly
    ds = ds[[v for v in fieldentryorder["vars"] if v in ds]]
    old_attrs = ds.attrs
    new_attrs = {k: old_attrs[k] for k in fieldentryorder["globattrs"] if k in old_attrs}

    # Add any remaining attributes at the end
    for k in old_attrs:
        if k not in new_attrs:
            new_attrs[k] = old_attrs[k]
    ds.attrs = new_attrs

    # Clean global attributes that are none-type
    ds.attrs = {k: ("" if v is None else v) for k, v in ds.attrs.items()}

    # Clean variable-level attributes that are none-type
    for var in ds.variables:
        ds[var].attrs = {k: ("" if v is None else v) for k, v in ds[var].attrs.items()}

    # Saving netCDF file
    #output_filename = fp.parent / f"{timestamp_clean}{serial_str}_Nph_all.nc"
    #output_filename = fp.parent / f"{timestamp_clean}_{serial_str}_{site_name_str}_{data_institution}_v2.0.TPB.nc"
    output_filename = fp.parent / f"{timestamp_clean}_{serial_str}_{site_name_str}_{data_institution}_v1.1.TPB.nc"
    ds.to_netcdf(output_filename)

    print(f"NetCDF file saved: {output_filename}")

def convertmattonetcdf(filepath, args):
    print(filepath)
    print("Error: The function convertmattonetcdf is not yet completed...")
    sys.exit(1)
    ds = None
    ds.attrs["source"] = "MAT File"

    # Modifying DS to fit template:
    ds = buildxrds(ds, args)

    # Clean global attributes that are none-type
    ds.attrs = {k: ("" if v is None else v) for k, v in ds.attrs.items()}

    # Clean variable-level attributes that are none-type
    for var in ds.variables:
        ds[var].attrs = {k: ("" if v is None else v) for k, v in ds[var].attrs.items()}

    fp = Path(filepath)
    output_filename = fp.with_suffix(".nc")
    ds.to_netcdf(output_filename)

    print(f"NetCDF file saved: {output_filename}")

def convertbufrtonetcdf(filepath, args):
    # First, find the site name from the initial lat/lon coordinates
    df_latlon = pdbufr.read_bufr(
        filepath,
        reader='generic',
        columns=['latitude','longitude'],
    )

    #print(f"Initial lat/lon from BUFR file: {df_latlon['latitude'].iloc[0]}, {df_latlon['longitude'].iloc[0]}")

    site_name = find_nearest_station(df_latlon['latitude'].iloc[0],
                                     df_latlon['longitude'].iloc[0],
                                     station_lookup,
                                     max_distance_km=15)[0]
    if site_name == None:
        print("Trying this file with a more relaxed distance threshold of 20 km...")
        time.sleep(1)
        site_name = find_nearest_station(df_latlon['latitude'].iloc[0],
                                         df_latlon['longitude'].iloc[0],
                                         station_lookup,
                                         max_distance_km=20)[0]

    # These are the profile columns which will contain all of the radiosonde data
    profile_columns = [
        'timePeriod',
        'pressure',
        'nonCoordinateGeopotentialHeight',
        'latitudeDisplacement',
        'longitudeDisplacement',
        'airTemperature',
        'dewpointTemperature',
        'windDirection',
        'windSpeed'
    ]

    # These are the metadata columns, specified for the Kolsass and Sterzing launch sites separately
    if site_name == "Sterzing":
        metadata_columns = [
            'radiosondeSerialNumber',
            'radiosondeType',
            'year', 'month', 'day', 'hour', 'minute', 'second',
            'numberOfSubsets',
            'latitude',
            'longitude',
            'edition',
            'masterTableNumber',
            'bufrHeaderCentre',
            'bufrHeaderSubCentre',
            'updateSequenceNumber',
            'dataCategory',
            'internationalDataSubCategory',
            'dataSubCategory',
            'masterTablesVersionNumber',
            'localTablesVersionNumber',
            'radiosondeType',
            'solarAndInfraredRadiationCorrection',
            'trackingTechniqueOrStatusOfSystem',
            'measuringEquipmentType',
            'radiosondeAscensionNumber',
            'correctionAlgorithmsForHumidityMeasurements',
            'radiosondeOperatingFrequency',
            'pressureSensorType',
            'temperatureSensorType',
            'humiditySensorType',
            'softwareVersionNumber',
            'timeSignificance',
            'heightOfStationGroundAboveMeanSeaLevel'
        ]

    elif site_name == "Kolsass":
        metadata_columns = [
            'radiosondeSerialNumber',
            'radiosondeType',
            'wigosIdentifierSeries',
            'wigosIssuerOfIdentifier',
            'wigosIssueNumber',
            'wigosLocalIdentifierCharacter',
            'year', 'month', 'day', 'hour', 'minute', 'second',
            'observerIdentification',
            'numberOfSubsets',
            'latitude',
            'longitude',
            'edition',
            'masterTableNumber',
            'bufrHeaderCentre',
            'bufrHeaderSubCentre',
            'updateSequenceNumber',
            'dataCategory',
            'internationalDataSubCategory',
            'dataSubCategory',
            'masterTablesVersionNumber',
            'localTablesVersionNumber',
            'radiosondeType',
            'solarAndInfraredRadiationCorrection',
            'trackingTechniqueOrStatusOfSystem',
            'measuringEquipmentType',
            'radiosondeAscensionNumber',
            'radiosondeReleaseNumber',
            'radiosondeConfiguration',
            'correctionAlgorithmsForHumidityMeasurements',
            'radiosondeGroundReceivingSystem',
            'radiosondeOperatingFrequency',
            'balloonManufacturer',
            'balloonType',
            'weightOfBalloon',
            'typeOfGasUsedInBalloon',
            'balloonFlightTrainLength',
            'pressureSensorType',
            'temperatureSensorType',
            'humiditySensorType',
            'radome',
            'softwareVersionNumber',
            'reasonForTermination',
            'timeSignificance',
            'heightOfStationGroundAboveMeanSeaLevel'
        ]

    elif site_name == "Bozen":
        metadata_columns = [
            'radiosondeSerialNumber',
            'radiosondeType',
            'wigosIdentifierSeries',
            'wigosIssuerOfIdentifier',
            'wigosIssueNumber',
            'wigosLocalIdentifierCharacter',
            'year', 'month', 'day', 'hour', 'minute', 'second',
            'observerIdentification',
            'numberOfSubsets',
            'latitude',
            'longitude',
            'edition',
            'masterTableNumber',
            'bufrHeaderCentre',
            'bufrHeaderSubCentre',
            'updateSequenceNumber',
            'dataCategory',
            'internationalDataSubCategory',
            'dataSubCategory',
            'masterTablesVersionNumber',
            'localTablesVersionNumber',
            'radiosondeType',
            'solarAndInfraredRadiationCorrection',
            'trackingTechniqueOrStatusOfSystem',
            'measuringEquipmentType',
            'radiosondeAscensionNumber',
            'radiosondeReleaseNumber',
            'radiosondeConfiguration',
            'correctionAlgorithmsForHumidityMeasurements',
            'radiosondeGroundReceivingSystem',
            'radiosondeOperatingFrequency',
            'balloonManufacturer',
            'balloonType',
            'weightOfBalloon',
            'typeOfGasUsedInBalloon',
            'balloonFlightTrainLength',
            'pressureSensorType',
            'temperatureSensorType',
            'humiditySensorType',
            'radome',
            'softwareVersionNumber',
            'reasonForTermination',
            'timeSignificance',
            'heightOfStationGroundAboveMeanSeaLevel'
        ]

    else:
        print(f"No launch site could be found from BUFR file lat/lon data. File {filepath} caused the error.")
        sys.exit(1)

    df_profile = pdbufr.read_bufr(
        filepath,
        reader='generic',
        columns=profile_columns,
    )

    if df_profile.empty:
        print("One of the columns for df_profile was not found in the BUFR file.")
        sys.exit(1)

    df_meta = pdbufr.read_bufr(
        filepath,
        reader='generic',
        columns=metadata_columns
    )

    if df_meta.empty:
        print("One of the columns for df_meta was not found in the BUFR file.")
        sys.exit(1)

    # Retrieve launch time
    y, m, d, H, M, S = df_meta[['year', 'month', 'day', 'hour', 'minute', 'second']].iloc[0]
    launch_time = pd.Timestamp(year=y, month=m, day=d, hour=H, minute=M, second=S, tz='UTC')

    # Create datetime coordinate by using the launch time and timePeriod (in s since launch)
    # then sort in ascending order and average where there are multiple values to ensure monotonicity
    df_profile["datetime"] = pd.to_datetime(launch_time + pd.to_timedelta(df_profile["timePeriod"], unit="s"), utc=True).dt.tz_localize(None)
    df_profile = (df_profile.sort_values("datetime").groupby("datetime", as_index=False).mean(numeric_only=True))
    df_profile = df_profile.set_index("datetime")

    """# Note: I have implicitly assumed that the primary dimension is monotonically increasing time in seconds
    # This will be noted as a comment in the file for future reference
    ds = xr.Dataset()
    for col in profile_columns:
        ds[col] = ('Time', df_profile[col].values)
    ds = ds.assign_coords(Time=range(len(df_profile)))"""

    ds = xr.Dataset.from_dataframe(df_profile)

    if ds.sizes["datetime"] < 5:
        fname = os.path.basename(filepath)
        print(f"Warning: {fname} has a launch that is too short (< 5 time points). Skipping.")
        return

    # Add metadata
    for col in metadata_columns:
        value = df_meta[col].iloc[0]  # scalar per sounding
        ds.attrs[col] = value

    # Rename latitude and longitude attributes
    ds.attrs["startLatitude"] = ds.attrs.pop("latitude")
    ds.attrs["startLongitude"] = ds.attrs.pop("longitude")

    # Create new latitude and longitude arrays
    ds['lat'] = df_latlon['latitude'].iloc[0] + ds["latitudeDisplacement"]
    ds['lon'] = df_latlon['longitude'].iloc[0] + ds["longitudeDisplacement"]

    ds.attrs['launch_time'] = launch_time.isoformat()
    ds.attrs["site_name"] = site_name
    ds.attrs["source"] = "BUFR File"

    # Modifying DS to fit template:
    ds = buildxrds(ds, args)

    # Adding administrative attributes
    ds.attrs["Conventions"] = "CF-1.13"
    ds.attrs["processing_level"] = np.int32(3)
    ds.attrs["project"] = "Multi-scale transport and exchange processes in the atmosphere over mountains – programme and experiment (TEAMx)"
    ds.attrs["featureType"] = "trajectory"
    ds.attrs["originalFile"] = os.path.basename(filepath)

    # Setting filename strings and acknowledgements
    timestamp_clean = pd.Timestamp(ds.attrs['launch_time']).strftime("%Y%m%d%H%M")
    serial_str = ds.attrs.get("serial", "UNKNOWN").strip()
    site_name_str = ds.attrs.get("site_name", "UNKNOWN").strip().replace(" ", "").upper()
    if site_name_str == "STERZING":
        data_institution = "NCAS"
        ds.attrs["acknowledgement"] = "Original Level 1 data provided by NCAS, UK"
    elif site_name_str == "KOLSASS":
        data_institution = "UIBK"
        ds.attrs["acknowledgement"] = "Original Level 1 data provided by UIBK, AT"
    elif site_name_str == "BOZEN":
        data_institution = "KIT_"
        ds.attrs["acknowledgement"] = "Original Level 1 data provided by KIT, DE"
    site_name_str = site_name_str.ljust(8, "_") # Pad to 8 characters for consistent filename formatting

    # Re-order variables and attributes sensibly
    ds = ds[[v for v in fieldentryorder["vars"] if v in ds]]
    old_attrs = ds.attrs
    new_attrs = {k: old_attrs[k] for k in fieldentryorder["globattrs"] if k in old_attrs}

    # Add any remaining attributes at the end
    for k in old_attrs:
        if k not in new_attrs:
            new_attrs[k] = old_attrs[k]
    ds.attrs = new_attrs

    # Clean global attributes that are none-type
    ds.attrs = {k: ("" if v is None else v) for k, v in ds.attrs.items()}

    # Clean variable-level attributes that are none-type
    for var in ds.variables:
        ds[var].attrs = {k: ("" if v is None else v) for k, v in ds[var].attrs.items()}

    # Saving netCDF file
    fp = Path(filepath)
    #output_filename = fp.with_suffix(".nc")
    if data_institution != "NCAS":
        output_filename = fp.parent / f"{timestamp_clean}_{serial_str}_{site_name_str}_{data_institution}_v2.0.TPB.nc"
    else:
        output_filename = fp.parent / f"{timestamp_clean}_{serial_str}_{site_name_str}_{data_institution}_v1.1.TPB.nc"
    ds.to_netcdf(output_filename)

    print(f"NetCDF file saved: {output_filename}")

def convertsondehubtonetcdf(snumber, args):
    """
    Fetch the sondehub data and create netcdf

    Parameters:
        snumber : Radiosonde Serial Number
    """
    
    # Download radiosonde data from SondeHub
    radiosonde = sondehub.download(serial=snumber)

    # Store as and fix pandas dataframe
    df = pd.DataFrame(radiosonde)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True).dt.tz_localize(None)
    df = df.set_index("datetime")
    global_attrs = {}
    non_constant_strs = []
    string_cols = df.select_dtypes(include="object").columns

    # Sort out string variables
    for col in string_cols:
        non_null = df[col].dropna().unique()
        if len(non_null) == 1:
            val = non_null[0]
            global_attrs[col] = str(val)
        else:
            non_constant_strs.append(col)

    df = df.drop(columns=global_attrs.keys())

    for col in non_constant_strs:
            df = df.drop(columns=col)

    # Create XArray Dataset
    ds = xr.Dataset.from_dataframe(df)
    ds.attrs.update(global_attrs)
    ds.attrs["source"] = "SondeHub"
    ds = buildxrds(ds, args)

    # Export to netCDF4
    os.chdir(data_dir)
    filename = "RS_data_{0}_{1}.nc".format(pd.to_datetime(ds['datetime'][0].values).strftime('%Y%m%d_%H%M%S'), global_attrs['serial'])
    ds.to_netcdf(filename)
    print(f'Created file {filename} in {os.path.abspath(data_dir)}')
    
    return ds

def buildxrds(ds, args):
    """
    Build the xarray dataset object, with the correct attributes and variable descriptors

    Parameters:
        ds : xr.Dataset
    """

    # Choose either Summer or Winter EOP
    dt = ds["datetime"].values[0]
    month = int(str(dt)[5:7])

    if month in [11, 12, 1, 2, 3]:
        season = "Winter "
    elif month in [5, 6, 7, 8, 9]:
        season = "Summer "
    else:
        season = ""  

    # General attributes
    ds.attrs["title"] = f"Radiosonde Dataset for {season}2025 TEAMx Campaign"
    ds.attrs["startdate"] = pd.to_datetime(ds['datetime'][0].values).strftime('%Y-%m-%d')
    ds.attrs["starttime"] = pd.to_datetime(ds['datetime'][0].values).strftime('%H:%M:%S UTC')
    ds.attrs["history"] = f"Created with xarray {xr.__version__} and pandas {pd.__version__}"
    ds.attrs["institution"] = "University of Bath, Claverton Down, Bath, BA2 7AY, United Kingdom"
    ds.attrs["date_created"] = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    ds.attrs["contact"] = "T. P. Banyard, tpb38@bath.ac.uk"
    if args.launch_site:
        ds = set_site_name(ds, force_site_name=args.launch_site)
    else:
        ds = set_site_name(ds)

    # Rename serradiosondeSerialNumberial to serial if it exists
    if "radiosondeSerialNumber" in ds.attrs:
        ds.attrs["serial"] = ds.attrs.pop("radiosondeSerialNumber")

    # Rename nonCoordinateGeopotentialHeight to alt if it exists
    if "nonCoordinateGeopotentialHeight" in ds:
        ds = ds.rename({"nonCoordinateGeopotentialHeight": "alt"})
        ds["alt"].attrs.update({
            "long_name": "Geopotential height above mean sea level",
            "units": "m"
        })

    # Find index of maximum (burst) altitude
    idx_burst = int(ds["alt"].argmax().values)
    max_alt = float(ds["alt"].isel(datetime=idx_burst).values)
    # Trim dataset to include only ascent (up to burst point)
    ds = ds.isel(datetime=slice(0, idx_burst + 1))

    # Compute range of radiosonde from launch site
    if ds.attrs['site_name'] != 'Unknown':
        name_lookup = {v["name"]: v for v in station_lookup.values()}
        lat0 = name_lookup[ds.attrs['site_name']]["lat"]
        lon0 = name_lookup[ds.attrs['site_name']]["lon"]
        R = 6371000.0 # Earth's radius (m)
        lat1 = np.radians(lat0)
        lon1 = np.radians(lon0)
        lat2 = np.radians(ds["lat"])
        lon2 = np.radians(ds["lon"])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
        c = 2 * np.arcsin(np.sqrt(a))
        distance = R * c  # great-circle distance in meters
        ds["CompRng"] = (("datetime",), distance.data)
        ds["CompRng"].attrs.update({
            "long_name": "Range of Radiosonde from Launch Site",
            "units": "m"
        })

        # Compute bearing of radiosonde from launch site
        x = np.sin(dlon) * np.cos(lat2)
        y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
        azimuth = (np.degrees(np.arctan2(x, y)) + 360) % 360
        ds["CompAz"] = (("datetime",), azimuth.data)
        ds["CompAz"].attrs.update({
            "long_name": "Azimuth from Launch Site",
            "units": "degrees clockwise from north"
        })

    # Make sure both heading and dir are in the netcdf file
    if "heading" in ds:
        # Calculate dir (wind direction)
        ds["windDir"] = (ds["heading"] + 180) % 360
        ds["windDir"].attrs.update({
            "_FillValue": ds["heading"].attrs.get("_FillValue", np.nan),
            "units": "degrees clockwise from north",
            "name": "Wind Direction (opposite of heading)"
        })

    elif "windDirection" in ds:
        # Calculate heading from wind direction
        ds["heading"] = (ds["windDirection"] + 180) % 360

        # Rename windDirection to windDir
        ds = ds.rename({"windDirection": "windDir"})

        # Update attributes for the renamed variable
        ds["heading"].attrs.update({
            "_FillValue": ds["windDir"].attrs.get("_FillValue", np.nan),
            "units": "degrees clockwise from north",
            "name": "Heading (opposite of Wind Direction)"
        })

    else:
        print("Error: Neither 'heading' nor 'windDirection' found in dataset.")
        sys.exit(1)

    # Rename windSpeed to vel_h if it exists
    if "windSpeed" in ds:
        ds = ds.rename({"windSpeed": "vel_h"})

    # Compute zonal wind
    ds['u'] = -ds['vel_h'] * np.sin(np.deg2rad(ds['windDir']))
    ds["u"].attrs.update({
        "name": "Zonal wind",
        "units": "m s-1",
        "standard_name": "eastward_wind"
    })

    # Compute meridional wind
    ds["v"] = -ds["vel_h"] * np.cos(np.deg2rad(ds["windDir"]))
    ds["v"].attrs.update({
        "name": "Meridional wind",
        "units": "m s-1",
        "standard_name": "northward_wind"
    })
    
    # Convert pressure to hPa if it is in Pa to start with
    if "pressure" in ds:
        if float(ds["pressure"].median()) > 2000:
            ds["pressure"] = ds["pressure"] / 100.0 # Convert to hPa
    else:
        print("Pressure not in original file. Setting variable to NaNs.")
        ds["pressure"] = (("datetime",), np.full_like(ds["datetime"], np.nan, dtype=float))

    # Add vel_z if it does not exist
    if "vel_z" not in ds:
        ds["vel_z"] = ds["alt"].diff("datetime") / ds["timePeriod"].diff("datetime")
        ds["vel_z"] = ds["vel_z"].interpolate_na(dim="datetime")
        ds["vel_z"].attrs.update({
            "standard_name": "upward_balloon_velocity",
            "description": "Computed as d(altitude)/d(timePeriod)"
        })

    # Compute vertical wind parameters
    mask = ds["vel_z"].isnull() # Where is vel_z NaN?
    vel_z_nonans = ds["vel_z"].interpolate_na(dim="datetime") # Interpolate NaNs for Savitzky-Golay Filtering
    try:
        vel_z_smoothed = savgol_filter(vel_z_nonans, window_length=301, polyorder=2) # Run SG Filter to get large-scale ascent
        vel_z_prime = vel_z_nonans - vel_z_smoothed # Obtain perturbation by subtracting smoothed from original
        vel_z_prime = vel_z_prime.where(~mask) # Put NaNs back where vel_z was originally NaN

        # Obtain smoothed vertical wind to omit turbulence and sensor noise by applying rolling mean to the perturbations
        vel_z_prime_nonans = fixnanswithmean(vel_z_prime) # Replace NaNs with mean for rolling mean calculation

    except:
        print("Error applying Savitzky-Golay filter. Setting vel_z_smoothed and vel_z_prime to NaNs.")
        vel_z_smoothed = np.full_like(ds["vel_z"], np.nan, dtype=float)
        vel_z_prime = np.full_like(ds["vel_z"], np.nan, dtype=float)
        vel_z_prime_nonans = vel_z_nonans

    turb_window, noise_window = 21, 3 # Window sizes for turbulence and noise (must be odd)
    vel_z_prime_smoothed = vel_z_prime_nonans.rolling({"datetime": turb_window}, center=True, min_periods=1).mean()
    vel_z_prime_smoothed_twice = vel_z_prime_smoothed.rolling({"datetime": noise_window}, center=True, min_periods=1).mean()
    vel_z_prime_smoothed = vel_z_prime_smoothed.where(~mask)
    vel_z_prime_smoothed_twice = vel_z_prime_smoothed_twice.where(~mask)

    # Create variables for the smoothed vertical wind and the perturbations
    ds["vel_z_smoothed"] = (("datetime",), vel_z_smoothed.data)
    ds["vel_z_smoothed"].attrs.update({
            "long_name": "Smoothed balloon ascent rate",
            "description": "Represents the large-scale ascent attributed to the balloon buoyancy. Computed using a Savitzky-Golay filter (window=301s, polyorder=2).",
            "units": "m s-1"
        })
    ds["vel_z_prime"] = (("datetime",), vel_z_prime.data)
    ds["vel_z_prime"].attrs.update({
            "long_name": "Vertical velocity perturbation",
            "description": "Represents the vertical wind velocity attributed to both sensor noise + turbulence + gravity waves. vel_z = vel_z_smoothed + vel_z_prime.",
            "units": "m s-1"
        })
    ds["vel_z_prime_smoothed"] = (("datetime",), vel_z_prime_smoothed_twice.data)
    ds["vel_z_prime_smoothed"].attrs.update({
            "long_name": "Smoothed vertical velocity perturbation",
            "description": "Represents the vertical wind velocity attributed to gravity waves. Small-scale turbulence has been removed using a rolling mean with ~100 m vertical window, followed by sensor noise removed using a rolling mean with a ~15 m vertical window.",
            "units": "m s-1"
        })

    # Rename airTemperature to temp if it exists
    if "airTemperature" in ds:
        ds = ds.rename({"airTemperature": "temp"})
        # Code to check that ds["temp"] is in K for CF compliance
        if float(ds["temp"].mean()) > 100:
            ds["temp"] = ds["temp"] # Keep as K
        else:
            ds["temp"] = ds["temp"] + 273.15 # Convert to K
    
    # Rename dewpointTemperature to dewp if it exists
    if "dewpointTemperature" in ds:
        ds = ds.rename({"dewpointTemperature": "dewp"})
        # Code to check that ds["dewp"] is in K for CF compliance
        if float(ds["dewp"].mean()) > 100:
            ds["dewp"] = ds["dewp"] # Keep as K
        else:
            ds["dewp"] = ds["dewp"] + 273.15 # Convert to K

    # ---------------------------------------------------------------------------
    # Potential temperature (θ) for dry air:
    #
    #     θ = T * (p_ref / p) ** (R_d / c_p)
    #
    # where:
    #     θ     = potential temperature (K)
    #     T     = actual temperature (K)
    #     p     = actual pressure (Pa)
    #     p_ref = reference pressure (100000 Pa = 1000 hPa)
    #     R_d   = gas constant for dry air (287 J/kg/K)
    #     c_p   = specific heat of dry air at constant pressure (1004 J/kg/K)
    #
    # Reference: Holton, J. R. (2004), An Introduction to Dynamic Meteorology, 4th Edition, p.50
    # ---------------------------------------------------------------------------

    # Compute potential temperature (in SI units, not dataset units)
    T = ds["temp"]  # keep as K
    p = ds["pressure"] * 100.0  # hPa → Pa
    p_ref = 1000.0 * 100.0  # 1000 hPa in Pa
    R_d = 287.0 # gas constant for dry air (J/kg/K)
    c_p = 1004.0 # specific heat of dry air (J/kg/K)
    theta = T * (p_ref / p) ** (R_d / c_p)
    ds["PotTemp"] = (("datetime",), theta.data)
    ds["PotTemp"].attrs.update({
        "long_name": "Potential Temperature",
        "units": "K",
        "standard_name": "air_potential_temperature"
    })

    # ---------------------------------------------------------------------------
    # Specific humidity (q)
    #
    #     q = (ε * e) / (p - (1 - ε) * e)
    #
    # where:
    #     q = specific humidity (kg/kg)
    #       = m_v / (m_v + m_d) (using the Ideal Gas Law pV=nRT where n=m/M)
    #       = εe / p - (1 - ε)e (and substituting for m_v and m_d..........)
    #     e = water vapour partial pressure (Pa)
    #     p = total air pressure (Pa)
    #     ε = ratio of molecular masses of water vapour and dry air
    #       = M_v / M_d = 18.016 / 28.966 = 0.622
    #
    # Water vapour pressure (e) is estimated from temperature and relative humidity:
    #
    #     e_sat = 6.112 * exp[(17.67 * (T_C)) / (T_C + 243.5)]       # saturation vapour pressure (hPa)
    #     e     = (RH / 100) * e_sat                                 # vapour pressure (hPa)
    #
    # where:
    #     T_C = air temperature in °C  = (T_K - 273.15)
    #     RH  = relative humidity in %
    #
    # References: 
    #     Ambaum, M. H. P. (2010), Thermal Physics of the Atmosphere, p.98
    #     Salby, M. L. (1996), Fundamentals of Atmospheric Physics, p.138
    # ---------------------------------------------------------------------------

    # Compute humidity if it does not exist
    T_C = ds["temp"] - 273.15  # convert K → °C
    Td_C = ds["dewp"] - 273.15  # convert K → °C
    e_sat = 6.112 * np.exp((17.67 * T_C) / (T_C + 243.5))
    if "relative_humidity" not in ds:
        if "dewp" in ds:
            e  = 6.112 * np.exp((17.67 * Td_C) / (Td_C + 243.5))
            RH = (e / e_sat) * 100
            ds["relative_humidity"] = (("datetime",), RH.data)
            ds["relative_humidity"].attrs.update({
                "long_name": "Relative Humidity",
                "units": "%"
            })
        else:
            print("Relative humidity not in original file. Setting variable to NaNs.")
            ds["relative_humidity"] = (("datetime",), np.full_like(ds["datetime"], np.nan, dtype=float))
            print("Dewpoint not in original file. Setting variable to NaNs.")
            ds["dewp"] = (("datetime",), np.full_like(ds["datetime"], np.nan, dtype=float))

    # Compute specific humidity
    e = ds["relative_humidity"] / 100.0 * e_sat # vapor pressure (hPa)
    e_Pa = e * 100.0
    p_Pa = ds["pressure"] * 100.0
    q = 0.622 * e_Pa / (p_Pa - 0.378 * e_Pa) # specific humidity, q (kg/kg)
    ds["specific_humidity"] = (("datetime",), q.data)
    ds["specific_humidity"].attrs.update({
        "long_name": "Specific Humidity",
        "units": "kg kg-1",
        "standard_name": "specific_humidity"
    })

    # ---------------------------------------------------------------------------
    # Dew point (T_d) based on Vaisala documentation (B210973EN-F)
    #
    #     P_ws = A * 10^(m * T / (T + T_n))      # saturation vapour pressure (hPa)
    #     P_w  = (RH / 100) * P_ws               # vapour pressure (hPa)
    #     T_d  = (T_n * log10(P_w / A)) / (m - log10(P_w / A))
    #
    # Constants (for water, valid roughly -20°C to +50°C):
    #     A  = 6.116441  hPa
    #     m  = 7.591386  (dimensionless)
    #     Tn = 240.7263  °C
    #
    # Reference:
    #     Vaisala, "Humidity Conversion Formulas", Technical eBook B210973EN-F (2021)
    # ---------------------------------------------------------------------------

    # Compute dewpoint if it does not exist
    if "dewp" not in ds:
        # Clean up RH by always selecting the minimum RH
        ds["relative_humidity"] = ds["relative_humidity"].rolling(datetime=30, min_periods=1).reduce(np.nanmin)

        # Compute dewpoint and get description
        dewp, desc = dew_point_from_RH(
            T_C,
            ds["relative_humidity"],
            ds["pressure"],
            method=args.dewpoint_method
        )

        # Add to dataset
        ds["dewp"] = (("datetime",), dewp.data + 273.15) # Convert to K for CF compliance
        ds["dewp"].attrs.update({
            "units": "K",
            "description": desc
        })

    # Make sure that there is only data where the alt is not NaN.
    # Data must be paired with a valid altitude.
    # Pad by 20 elements either side
    # ds = ds.where(np.isfinite(ds["alt"]))
    pad = 20
    nan_mask = np.isnan(ds["alt"])
    for shift in range(-pad, pad + 1):
        nan_mask |= np.isnan(np.roll(ds["alt"], shift))
    valid_mask = ~nan_mask
    ds = ds.where(valid_mask)

    # Mask the top 10 ten points of the profile in case it contains bad data
    mask = np.isfinite(ds["alt"])
    n = 10
    if mask.sum() > n:  # only if we have enough valid points
        last_valid_indices = np.where(mask)[0][-n:]
        mask[last_valid_indices] = False
    ds = ds.where(mask)

    # Apply attributes if variable exists in dataset
    for var, attrs in metadata.items():
        if var in ds:
            ds[var].attrs.update(attrs)

    # Rename datetime to time for CF compliance
    ds = ds.rename({"datetime": "time"})
    ds["time"].attrs.update({
        "standard_name": "time",
        "axis": "T",
        "long_name": "Time"
    })

    # Sort out qc_flag variable if it exists
    if "qc_flag" in ds:
        ds["qc_flag"] = ds["qc_flag"].fillna(-1).astype("int64")
        ds["qc_flag"].attrs.update({
                "_FillValue": -1,
                "flag_values": np.array([0,1,2,3,4,5,6,7,8,9], dtype=np.int32),
                "flag_meanings": (
                "0 - not_used; "
                "1 - good_data; "
                "2 - first_point_no_ascent_rate; "
                "3 - suspect_data_zero_ascent_rate; "
                "4 - suspect_data_zero_wind_speed; "
                "5 - time_interval_<_1s_resolution; "
                "6 - time_change_not_captured_by_1s_resolution; "
                "7 - ascent_rate_not_calculable; "
                "8 - wind_or_position_data_missing; "
                "9 - ptu_data_missing"
            )
        })

    return ds

def main(args):
    # Keep track of created files
    created_files = []

    # Process given serial numbers
    if args.serial_numbers:
        for snumber in args.serial_numbers:
            # Download radiosonde data from SondeHub
            print(f"Processing serial number: {snumber}")
            convertsondehubtonetcdf(snumber, args)

            created_files.append(snumber)

    # Process given files
    elif args.files:
        for f in args.files:
            if os.path.isdir(f):
                # Single directory
                for root, _, files in os.walk(f):
                    for file in files:
                        filepath = os.path.join(root, file)
                        # Processing radiosonde data from file
                        print(f"Processing file: {filepath}")
                        _, ext = os.path.splitext(filepath.strip()) # Extract file extension
                        ext = ext.lower().lstrip(".") # Remove dot and make lowercase
                        if ext in ["bufr", "bfr"]:
                            convertbufrtonetcdf(filepath, args)
                        elif ext in ["mat", "m"]:
                            convertmattonetcdf(filepath, args)
                        elif ext == "nc":
                            convertnphnctonetcdf(filepath, args)
                        else:
                            print("Unknown file type.")

                        created_files.append(filepath)
            else:
                # Processing radiosonde data from file
                print(f"Processing file: {f}")
                _, ext = os.path.splitext(f.strip()) # Extract file extension
                ext = ext.lower().lstrip(".") # Remove dot and make lowercase
                if ext in ["bufr", "bfr"]:
                    convertbufrtonetcdf(f, args)
                elif ext in ["mat", "m"]:
                    convertmattonetcdf(f, args)
                elif ext == "nc":
                    fname = os.path.basename(f)
                    if fname.startswith("Sonde"):
                        convertnphnctonetcdf(f, args)
                    elif fname.startswith("ncas-radiosonde"):
                        convertprnctonetcdf(f, args)
                else:
                    print("Unknown file type.")

                created_files.append(f)
    
    # Print message to confirm imposed launch site name
    if args.launch_site:
        print(f"Launch site forced to: {args.launch_site}")

    # Success message depending on how many were processed
    if len(created_files) == 1:
        print("Radiosonde netCDF file created successfully.")
    elif len(created_files) > 1:
        print("Radiosonde netCDF files created successfully.")
    else:
        print("No files created.") # Should not normally happen

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Radiosonde NetCDF File Creator"
    )

    group = parser.add_mutually_exclusive_group(required=True)

    group.add_argument(
        '-n', '--serial-numbers',
        nargs='+',
        help='One or more radiosonde serial number(s) to download from SondeHub'
    )

    group.add_argument(
        '-f', '--files',
        nargs='+',
        help='One or more local file paths or directories'
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