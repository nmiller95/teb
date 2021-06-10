[![astropy](http://img.shields.io/badge/powered%20by-AstroPy-orange.svg?style=flat)](http://www.astropy.org/) 
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# teb: Temperatures for Eclipsing Binary stars

teb is a Python package that calculates fundamental effective temperatures for solar-type stars in eclipsing binary systems using photometry, Gaia parallax and radii. The full method is described in [Miller, Maxted & Smalley (2020)](https://ui.adsabs.harvard.edu/abs/2020MNRAS.497.2899M/abstract).

## Installation

Clone the repository .

```bash
$ git clone https://github.com/nmiller95/teb.git
```

## Requirements

Developed in anaconda python 3.7

- [astropy](https://pypi.org/project/astropy/) 4.2.1 or later
- [synphot](https://pypi.org/project/synphot/) 1.0.1 or later
- [astroquery](https://pypi.org/project/astroquery/) 0.4.2 or later
- [pyvo](https://pypi.org/project/pyvo/) 1.1 or later
- [emcee](https://pypi.org/project/emcee/) 3.0.2 or later
- [corner](https://pypi.org/project/corner/) 2.2.1 or later
- [uncertainties](https://pypi.org/project/uncertainties/) 3.1.5 or later

Plus standard libraries: matplotlib, scipy, numpy, yaml, cpickle, os, warnings


## Usage

Set up the configuration yaml files with your parameters then run the main script.

```bash
$ python3 teb.py
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
