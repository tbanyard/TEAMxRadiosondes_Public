def read_config_file(config_file):
    """Reads a simple key-value config file and turns it into a dict.

    Parameters
    ----------
    config_file : str
        Path to the config file.
        Lines beginning with '#' are treated as comments and ignored.

    Returns
    -------
    dict
        Dictionary mapping each key (str) to its value (str), with any
        surrounding single or double quotes stripped.
    """
    config = {}
    with open(config_file, 'r') as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            key, value = line.split(None, 1)
            config[key] = value.strip("'\"")
    return config