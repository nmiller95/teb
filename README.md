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

Set up the data file for your star

```bash
 python3 teb.py --make-file  [star_name]
```
where [star_name] is a name resolvable by SIMBAD or the coordinates of the
star in the form "Jhhhmmmss.s+ddmmss.s". The output file
config/[star_name].yaml can be re-named to a more convenient name for the star
once it is created.

See ..
```bash
  python3 teb.py --help
```
 .. for help with this step.

Add / adjust observed data for your star in the resulting file
config/[star_name].yaml

Set the star name and other parameters for the analysis in config/config.yaml

Then run your analysis using
```bash
  python3 teb.py 
```

Use the python scripts in the folder "scripts" for plotting your results, and
to generate output LaTeX tables.

See the usage instructions in calspec.py if you need to use new filters not
already included in cache/fps. The file calspec/calspec_lovar.csv can be used
with the multi-cone VO service in topcat to collect photometric data for
this sample CALSPEC stars excluding those with known variability > 0.5%. 
(topcat - https://www.star.bristol.ac.uk/mbt/topcat/)
 
To add filters used for flux ratio measurements only, use calspec.py with
the option "-n". 

For further details, see the built-in help for calspec:

```bash
  python3 calspec.py --help
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
