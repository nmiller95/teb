[![astropy](http://img.shields.io/badge/powered%20by-AstroPy-orange.svg?style=flat)](http://www.astropy.org/) 
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# teb: Temperatures for Eclipsing Binary stars

teb is a Python package that calculates fundamental effective temperatures for solar-type stars in eclipsing binary systems using photometry, Gaia parallax and radii. The full method is described in [Miller, Maxted & Smalley (2020)](https://ui.adsabs.harvard.edu/abs/2020MNRAS.497.2899M/abstract).

## Installation

Clone the repository:

```bash
$ git clone https://github.com/nmiller95/teb.git
```
Then use pip to install the requirements.txt file
```bash
$ python3 -m pip install -r requirements.txt
```

teb was developed in Python 3.7. 

## Usage

Set up the configuration yaml files with your parameters then run the main script.

```bash
$ python3 teb.py -c <configfile> -p <photometryfile> -f <frpfile>
```

To save the command line output to a log file for longer runs, use

```bash
$ nohup python3 teb.py -c <configfile> -p <photometryfile> -f <frpfile> > & output.log &
```

You can make an empty set of configuration files using the command
```bash
$ python3 teb.py -m <newfile>
```

Full details of the usage can be found with the help command
```bash
$ python3 teb.py --help
```

### A note on stellar models

Two sources of model SED are supported in `teb`. 

- [BT-Settl](http://svo2.cab.inta-csic.es/theory/newov2/index.php?models=bt-settl)
  - Detailed logg-Teff coverage
  - [Asplund et al (2009)](https://ui.adsabs.harvard.edu/abs/2009ARA%26A..47..481A/abstract) abundances
  - \[M/H\] from 0.5 to -4.0 supported
- [BT-Settl-CIFIST](http://svo2.cab.inta-csic.es/theory/newov2/index.php?models=bt-settl-cifist)
  - Detailed logg-Teff coverage
  - [Caffau et al (2011)](https://ui.adsabs.harvard.edu/abs/2011SoPh..268..255C/abstract) abundances
  - Only \[M/H\] = 0.0 supported

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
