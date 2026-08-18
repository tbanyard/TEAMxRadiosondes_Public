import numpy as np

def haversine(lats, lons):
    """Returns the distance between two lat/lon points in kilometers.

    Parameters
    ----------
    lats : array-like of length 2
        Latitudes of the two points in degrees: [lat1, lat2].
    lons : array-like of length 2
        Longitudes of the two points in degrees: [lon1, lon2].

    Returns
    -------
    float
        Distance between the two points in kilometers.

    Notes
    -----
    - Inputs must be in degrees. Conversion to radians is handled internally.
    - Uses a spherical Earth approximation with radius 6371 km.

    Example
    --------
    >>> haversine([51.5, 48.9], [-0.1, 2.4])
    343.4  # approx distance London–Paris in km
    """
    
    R = 6371 # Earth's radius in km
    lat1, lat2 = np.radians(lats)
    lon1, lon2 = np.radians(lons)
    # Haversine formula
    a = (np.sin((lat2-lat1)/2) ** 2) + (np.cos(lat1) * \
        np.cos(lat2) * (np.sin((lon2-lon1)/2) ** 2))
    c = 2 * np.arcsin(np.sqrt(a))
    d = R * c
    return d