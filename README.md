<img src="https://github.com/schallerdavid/pyrod/blob/master/pyrod_logo.png" height="200">

PyRod is a python software developed to generate dynamic molecular interaction fields (dMIFs) and pharmacophore features based on analyzing the protein environment of water molecules in molecular dynamcis simulations.
Check the [PyRod wiki](https://github.com/schallerdavid/pyrod/wiki) for more information on installation and usage. The employed algorithms are described in the PyRod publication in [JCIM](https://doi.org/10.1021/acs.jcim.9b00281). 

## Installation
#### Clone this repository
Open a new terminal and clone this repository
```bash
git clone https://github.com/schallerdavid/pyrod.git ~/pyrod
```
#### Install dependencies
PyRod is written in Python 3.8 and relies on MDAnalysis (version >= 0.19.0, which includes NumPy and SciPy), as well as NumPy and SciPy themselves. All dependencies can be easily installed alongside PyRod using pip from within the PyRod directory:
```bash
cd PATH/TO/pyrod
```
```bash
pip install .
```
## Run PyRod
#### Load conda environment (optional)
```bash
source activate pyrod
```
#### Feed config file to pyrod
```bash
pyrod /path/to/pyrod_config_file.cfg
```
## Example
An example Jupyter Notebook for visualization of dMIFs using NGLView is provided in the [example directory](https://github.com/schallerdavid/pyrod/tree/master/example).

## Citation
Please cite our paper in [JCIM](https://doi.org/10.1021/acs.jcim.9b00281) when using PyRod for your research.

## License Information
PyRod is published under GNU General Public License v2.0. For more information read [LICENSE](https://github.com/schallerdavid/pyrod/blob/master/LICENSE).
