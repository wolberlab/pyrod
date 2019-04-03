<img src="https://github.com/schallerdavid/pyrod/blob/master/pyrod_logo.png" height="200">

PyRod is a python software developed to generate dynamic molecular interaction fields (dMIFs) and pharmacophore features based on analyzing the protein environment of water molecules in molecular dynamcis simulations.

## Installation
#### Clone this repository
Open a new terminal and clone this repository
```bash
git clone https://github.com/schallerdavid/pyrod.git ~/pyrod
```
#### Install dependencies
PyRod is written in python 3.6 and uses MDAnalysis (>= 0.19.0, is shipped with NumPy and SciPy), NumPy and SciPy which can be easily installed using pip:
```bash
pip3 install --upgrade MDAnalysis
```
or 
```bash
python3 -m pip install --upgrade MDAnalysis
```
You can also use conda to install all dependencies:
```bash
conda config --add conda-forge
conda create -n pyrod mdanalysis python=3.6
```
#### Create alias for your bash
```bash
echo 'alias pyrod="python ~/pyrod/pyrod.py"' >> ~/.bashrc
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
An example jupyter notebook for visualization of dmifs using nglview is provided in the example directory.
## License Information
PyRod is published under GNU General Public License v2.0. For more information read LICENSE.
