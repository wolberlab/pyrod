""" PyRod - dynamic molecular interaction fields (dMIFs), based on tracing water molecules in MD simulations.

This module contains helper functions to write out data.
"""


# python standard libraries
import xml.etree.ElementTree as et

# pyrod modules
try:
    from pyrod.modules.lookup import feature_names
    from pyrod.modules.helper_dmif import grid_parameters
    from pyrod.modules.helper_math import mean, standard_deviation
    from pyrod.modules.helper_update import update_user
except ImportError:
    from modules.lookup import feature_names
    from modules.helper_dmif import grid_parameters
    from modules.helper_math import mean, standard_deviation
    from modules.helper_update import update_user


def pml_reader(path):
    pharmacophore = et.parse(path)
    return pharmacophore