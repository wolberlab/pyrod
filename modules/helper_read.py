""" PyRod - dynamic molecular interaction fields (dMIFs), based on tracing water molecules in MD simulations.

This module contains helper functions to write out data.
"""


# python standard libraries
import numpy as np
import pickle
import xml.etree.ElementTree as et

# pyrod modules
try:
    from pyrod.modules.lookup import feature_names
    from pyrod.modules.helper_trajectory import grid_parameters
    from pyrod.modules.helper_math import mean, standard_deviation
    from pyrod.modules.helper_update import update_user
except ImportError:
    from modules.lookup import feature_names
    from modules.helper_trajectory import grid_parameters
    from modules.helper_math import mean, standard_deviation
    from modules.helper_update import update_user


def pml_reader(path):
    pharmacophore = et.parse(path)
    return pharmacophore


def pickle_reader(path, text, logger):
    update_user('Loading {} from {}.'.format(text, path), logger)
    with open(path, 'rb') as file:
        data = pickle.load(file)
    return data


def translate_pharmacophore(pharmacophore_path):
    """ This functions translates a LigandScout pharmacophore into an internal data format. """
    translate_dict = {'H': 'hi', 'PI': 'pi', 'NI': 'ni', 'AR': 'ai', 'HBD': 'hd', 'HBA': 'ha'}
    root = pml_reader(pharmacophore_path).getroot()
    features = root.findall('point') + root.findall('vector') + root.findall('plane')
    dtype = [('x', float), ('y', float), ('z', float), ('type', 'U10'), ('score', float), ('xi', float), ('yi', float),
             ('zi', float), ('tolerance_i', float)]
    feature_array = np.array([(0, 0, 0, '0', 0, 0, 0, 0, 0)] * len(features), dtype=dtype)
    for counter, feature in enumerate(features):
        feature_array['type'][counter] = translate_dict[feature.attrib['name']]
        feature_array['score'][counter] = float(feature.attrib['weight'])
        if feature.attrib['name'] not in ['HBD', 'HBA']:
            feature_array['x'][counter] = float(feature.find('position').attrib['x3'])
            feature_array['y'][counter] = float(feature.find('position').attrib['y3'])
            feature_array['z'][counter] = float(feature.find('position').attrib['z3'])
            if feature.attrib['name'] == 'AR':
                feature_array['xi'][counter] = float(feature.find('normal').attrib['x3'])
                feature_array['yi'][counter] = float(feature.find('normal').attrib['y3'])
                feature_array['zi'][counter] = float(feature.find('normal').attrib['z3'])
                feature_array['tolerance_i'][counter] = float(feature.find('normal').attrib['tolerance'])
        elif feature.attrib['name'] == 'HBA':
            feature_array['x'][counter] = float(feature.find('target').attrib['x3'])
            feature_array['y'][counter] = float(feature.find('target').attrib['y3'])
            feature_array['z'][counter] = float(feature.find('target').attrib['z3'])
            feature_array['xi'][counter] = float(feature.find('origin').attrib['x3'])
            feature_array['yi'][counter] = float(feature.find('origin').attrib['y3'])
            feature_array['zi'][counter] = float(feature.find('origin').attrib['z3'])
            feature_array['tolerance_i'][counter] = float(feature.find('origin').attrib['tolerance'])
        elif feature.attrib['name'] == 'HBD':
            feature_array['x'][counter] = float(feature.find('origin').attrib['x3'])
            feature_array['y'][counter] = float(feature.find('origin').attrib['y3'])
            feature_array['z'][counter] = float(feature.find('origin').attrib['z3'])
            feature_array['xi'][counter] = float(feature.find('target').attrib['x3'])
            feature_array['yi'][counter] = float(feature.find('target').attrib['y3'])
            feature_array['zi'][counter] = float(feature.find('target').attrib['z3'])
            feature_array['tolerance_i'][counter] = float(feature.find('target').attrib['tolerance'])
    return feature_array
