# TEAMxRadiosondes 🎈

> Tools for reading, processing, and analysing **radiosonde** data from the [TEAMx](https://www.teamx-programme.org/) campaign and related field projects.

---

## 🧭 Overview

**TEAMxRadiosondes** is a lightweight Python package designed for atmospheric scientists and meteorologists working with radiosonde soundings.  
It provides routines to:

- Read and convert **BUFR** radiosonde data into **NetCDF** and **xarray** datasets  
- Compute derived thermodynamic quantities (dew point, specific humidity, potential temperature, etc.)  
- Plot **Skew-T log-p** diagrams and other diagnostic visualisations  
- Interface easily with campaign datasets (TEAMx, SondeHub, Vaisala)

---

## 📦 Project Structure



TEAMxRadiosondes/
│
├── src/
│ └── TEAMxRadiosondes/
│ ├── init.py
│ ├── funcs/
│ │ ├── dry_adiabat.py
│ │ ├── moist_adiabat.py
│ │ ├── dewpoint.py
│ │ └── init.py
│ └── plotting/
│ ├── skewt.py
│ └── init.py
│
├── data/ # Example BUFR / NetCDF files
├── plots/ # Generated figures
├── pyproject.toml
├── setup.py
└── README.md


---

## ⚙️ Installation

You can install the package in **editable mode** during development:

```bash
# Create and activate a new environment
conda create -n teamx python=3.12
conda activate teamx

# Clone and install
git clone https://github.com/tbanyard/TEAMxRadiosondes.git
cd TEAMxRadiosondes
pip install -e .

🧪 Example Usage 1
import xarray as xr
from TEAMxRadiosondes.funcs import dry_adiabat, moist_adiabat

# Compute a dry adiabat starting at 25°C and 1000 hPa
Tdry, Pdry = dry_adiabat(25, 1000)

# Plot a Skew-T using a radiosonde dataset
from TEAMxRadiosondes.plotting import skewt
ds = xr.open_dataset("data/example_sounding.nc")
skewt.plot_skewt(ds)

🧪 Example Usage 2
# The below command will take existing radiosonde data files and convert them into 
# template netCDF files ready for use with this software
python src/TEAMxRadiosondes/createsondenetcdfs.py -f data/Sterzing/*.nc -l "Sterzing"

📊 Typical Output
<p align="center"> <img src="plots/example_skewt.png" width="400" alt="Example Skew-T plot"> </p>
🧩 Dependencies
Package	Version (tested)
numpy	2.0.1
matplotlib	3.10.0
xarray	2025.4.0
pandas	2.2.3
scipy	1.15.3
pdbufr	0.14.0
eccodes	2.43.0
netcdf4	1.7.2

(See environment.yml for a fully reproducible setup.)

📁 Data

The package expects input data in WMO BUFR or NetCDF format.
Utilities are included to convert between formats and harmonise metadata (launch time, serial number, etc.).

🧠 Citation

If you use this package in your research, please cite:

Banyard, T. P. (2025). TEAMxRadiosondes: Radiosonde Processing Toolkit for TEAMx and Related Campaigns. GitHub Repository. https://github.com/tbanyard/TEAMxRadiosondes

🧑‍💻 Development
# Run tests (if added later)
pytest tests/

# Format code
black src/TEAMxRadiosondes
ruff check src/

📜 License

This project is licensed under the MIT License.
See the LICENSE
 file for details.
