import pandas as pd
import xarray as xr
import numpy as np
import datetime
from TEAMxRadiosondes.funcs import set_site_name, fixnanswithmean, dew_point_from_RH
from scipy.signal import savgol_filter
import sys

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

def buildxrds(ds, args, skip_recomputation=False):
    """
    Build the radiosonde xarray dataset object, with the correct attributes and variable descriptors

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

    if not skip_recomputation:
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

    else:
        if "windDirection" in ds and "windDir" not in ds:
            ds = ds.rename({"windDirection": "windDir"})
    
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
    if not skip_recomputation:
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
    Td_C = ds["dewp"] - 273.15 if "dewp" in ds else None # convert K → °C
    e_sat = 6.112 * np.exp((17.67 * T_C) / (T_C + 243.5))
    if "relative_humidity" not in ds:
        if "specific_humidity" in ds:
            p_hPa = ds["pressure"]
            q = ds["specific_humidity"]
            e = q * p_hPa / (0.622 + 0.378 * q)
            RH = (e / e_sat) * 100
            ds["relative_humidity"] = (("datetime",), RH.data)
            ds["relative_humidity"].attrs.update({
                "long_name": "Relative Humidity",
                "units": "%"
            })
        elif "dewp" in ds:
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
    if not skip_recomputation:
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
