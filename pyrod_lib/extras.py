""" PyRod - dynamic molecular interaction fields (dMIFs), based on tracing water molecules in MD simulations.

Released under the GNU Public Licence v2.

This module contains additional functions meant for internal use.
"""

# python standard libraries
import pickle


# external libraries
import numpy as np


def dmif_equilibration(dmif_paths, feature_name, cutoff):
    with open(dmif_paths[0], 'rb') as file:
        dmif1 = pickle.load(file)
    differences = [0]
    for index in range(1, len(dmif_paths)):
        with open(dmif_paths[index], 'rb') as file:
            dmif2 = pickle.load(file)
        feature_difference = np.absolute(dmif1[feature_name] - dmif2[feature_name])
        differences.append(len(np.where(feature_difference > cutoff)[0]) / len(feature_difference))
    return differences
