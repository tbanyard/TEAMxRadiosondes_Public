from TEAMxRadiosondes.funcs import haversine

def find_nearest_station(lat, lon, station_lookup, max_distance_km=20):
    """Return the nearest station name and distance, or None if too far.
    
    Iterates over all stations in the lookup table, computes the
    haversine distance to each, and returns the closest one — provided
    it falls within the specified distance threshold.

    Parameters
    ----------
    lat : float
        Latitude of the query point in degrees.
    lon : float
        Longitude of the query point in degrees.
    station_lookup : dict
        Dictionary of stations, keyed by station code. Each value must
        contain at least 'lat', 'lon', and 'name' keys, e.g.:
            {"STR": {"lat": 46.88, "lon": 11.44, "name": "Sterzing"}}
    max_distance_km : float, optional
        Maximum allowable distance (km) for a match. If the nearest
        station exceeds this, None is returned. Default is 20 km.

    Returns
    -------
    nearest_station : str or None
        Name of the nearest station, or None if it exceeds the threshold.
    min_distance : float
        Distance to the nearest station in km, regardless of threshold.
        
    """
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