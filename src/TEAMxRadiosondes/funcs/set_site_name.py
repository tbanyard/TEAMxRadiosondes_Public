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