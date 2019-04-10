""" PyRod - dynamic molecular interaction fields (dMIFs), based on tracing water molecules in MD simulations.

Released under the GNU Public Licence v2.

This module contains functions to read data.
"""


# python standard libraries
import operator
import pickle
import sys
import xml.etree.ElementTree as et

# pyrod modules
try:
    from pyrod.pyrod_lib.lookup import feature_types
    from pyrod.pyrod_lib.math import mean, standard_deviation
    from pyrod.pyrod_lib.pharmacophore_helper import renumber_features
    from pyrod.pyrod_lib.write import update_user
except ImportError:
    from pyrod_lib.lookup import feature_types
    from pyrod_lib.math import mean, standard_deviation
    from pyrod_lib.pharmacophore_helper import renumber_features
    from pyrod_lib.write import update_user


def pickle_reader(path, text, logger):
    logger.info('Loading {} from {}.'.format(text, path))
    with open(path, 'rb') as file:
        data = pickle.load(file)
    return data


def pdb_pharmacophore_reader(path, pyrod_pharmacophore, logger):
    """ This functions reads pdb pharmacophores in pyrod format and stores them in an internal data structure. """
    if not pyrod_pharmacophore:
        update_user('This format is specific to pyrod pharmacophores. Make sure hte format is correct.', logger)
    pharmacophore = []
    with open(path, 'r') as pharmacophore_file:
        for line in pharmacophore_file.readlines():
            if line[:6].strip() == 'ATOM':
                if line[12:16].strip() == 'C':
                    feature_list = [0, 0, 0, 0, 0.0, [], 0.0, 0.0]
                    feature_list[0] = int(line[22:26])
                    feature_list[1] = line[17:20].strip()
                    feature_list[2] = line[21:22].strip()
                    feature_list[3] = [float(line[30:38]), float(line[38:46]), float(line[46:54])]
                    feature_list[4] = float(line[54:60])
                    feature_list[7] = float(line[60:66])
                    pharmacophore.append(feature_list)
                if 'P' in line[12:16]:
                    feature_list = [feature for feature in pharmacophore if feature[0] == int(line[22:26])]
                    pharmacophore = [feature for feature in pharmacophore if feature[0] == int(line[22:26])]
                    feature_list[5].append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
                    feature_list[6] = float(line[54:60])
                    pharmacophore.append(feature_list)
    return pharmacophore


def pml_feature_point_or_volume(feature_list, feature):
    """ This function adds point and volume feature attributes to the internal pharmacophore feature format. """
    feature_list[3] = [float(feature.find('position').attrib['x3']), float(feature.find('position').attrib['y3']),
                       float(feature.find('position').attrib['z3'])]
    feature_list[4] = float(feature.find('position').attrib['tolerance'])
    feature_list[7] = float(feature.attrib['weight'])
    return


def pml_feature_vector(feature_list, feature):
    """ This function adds vector feature attributes to the internal pharmacophore feature format. """
    feature_list[3] = [float(feature.find('origin').attrib['x3']), float(feature.find('origin').attrib['y3']),
                       float(feature.find('origin').attrib['z3'])]
    feature_list[4] = float(feature.find('origin').attrib['tolerance'])
    feature_list[5] = [[float(feature.find('target').attrib['x3']), float(feature.find('target').attrib['y3']),
                        float(feature.find('target').attrib['z3'])]]
    feature_list[6] = float(feature.find('target').attrib['tolerance'])
    if feature.attrib['name'] == 'HBA':  # switch to acceptor
        feature_list[3], feature_list[4], feature_list[5], feature_list[6] = [feature_list[5][0], feature_list[6],
                                                                              [feature_list[3]], feature_list[4]]
    if feature_list[1] in ['ha2', 'hd2', 'hda']:  # prepare format for correct partner assignment
        index = int(feature.attrib['featureId'].split('_')[2])
        if index == 0:
            feature_list[5] = [feature_list[5][0], []]
        elif index == 1:
            feature_list[5] = [[], feature_list[5][0]]
    feature_list[7] = float(feature.attrib['weight'])
    return


def pml_feature_plane(feature_list, feature):
    """ This function adds plane feature attributes to the internal pharmacophore feature format. """
    feature_list[3] = [float(feature.find('position').attrib['x3']), float(feature.find('position').attrib['y3']),
                       float(feature.find('position').attrib['z3'])]
    feature_list[4] = float(feature.find('position').attrib['tolerance'])
    feature_list[5] = [[float(feature.find('normal').attrib['x3']), float(feature.find('normal').attrib['y3']),
                        float(feature.find('normal').attrib['z3'])]]
    feature_list[6] = float(feature.find('normal').attrib['tolerance'])
    feature_list[7] = float(feature.attrib['weight'])
    return


def pml_feature(feature, pyrod_pharmacophore, logger):
    """ This function converts a pml feature into the internal pharmacophore feature format. """
    translate_features = {'H': 'hi', 'PI': 'pi', 'NI': 'ni', 'HBA': 'ha', 'HBD': 'hd', 'AR': 'ai', 'exclusion': 'ev'}
    feature_list = [0, 0, 0, 0, 0.0, [], 0.0, 0.0]
    feature_list[0] = feature.attrib['featureId']
    if feature.tag == 'volume':
        feature_list[1] = translate_features[feature.attrib['type']]
    else:
        feature_list[1] = translate_features[feature.attrib['name']]
    if pyrod_pharmacophore:
        try:
            feature_list[0] = int(feature.attrib['featureId'].split('_')[1])
            feature_list[1] = feature.attrib['featureId'].split('_')[0]
        except (IndexError, ValueError):
            update_user('You attempted to read a pml pharmacophore, that was not generated by PyRod. Please set pyrod '
                        'pharmacophore to false!', logger)
            sys.exit()
    feature_list[2] = 'M'
    if feature.attrib['optional'] == 'true':
        feature_list[2] = 'O'
    if feature.tag in ['point', 'volume']:
        pml_feature_point_or_volume(feature_list, feature)
    elif feature.tag == 'vector':
        pml_feature_vector(feature_list, feature)
    elif feature.tag == 'plane':
        pml_feature_plane(feature_list, feature)
    return feature_list


def merge_pyrod_hydrogen_bonds(pharmacophore, logger):
    """ This function merges hydrogen bond acceptor and donor features into the pyrod features ha2, hd2 and hda. """
    valid_features = [feature for feature in pharmacophore if feature[1] not in ['ha2', 'hd2', 'hda', 'ev']]
    exclusion_volumes = [feature for feature in pharmacophore if feature[1] == 'ev']
    pyrod_hydrogen_bonds = [feature for feature in pharmacophore if feature[1] in ['ha2', 'hd2', 'hda']]
    pyrod_hydrogen_bond_ids = set([feature[0] for feature in pyrod_hydrogen_bonds])
    for index in pyrod_hydrogen_bond_ids:
        feature_pair = [feature for feature in pyrod_hydrogen_bonds if feature[0] == index]
        pyrod_hydrogen_bond = feature_pair[0]
        try:
            if len(pyrod_hydrogen_bond[5][0]) == 0:
                pyrod_hydrogen_bond[5][0] = feature_pair[1][5][0]
            elif len(pyrod_hydrogen_bond[5][1]) == 0:
                pyrod_hydrogen_bond[5][1] = feature_pair[1][5][1]
            valid_features.append(pyrod_hydrogen_bond)
        except IndexError:
            update_user('The given pharmacophore contains incomplete hydrogen bonding features. Feature type is {} '
                        'but only one interaction partner was found. Either create a new pharmacophore without '
                        'splitting ha2, hd2 and hda features or set pyrod pharmacophore to false.'. format(
                         pyrod_hydrogen_bond[1]), logger)
            sys.exit()
    valid_features.sort(key=operator.itemgetter(0))
    return valid_features + exclusion_volumes


def pml_pharmacophore_reader(path, pyrod_pharmacophore, logger):
    """ This functions reads LigandScout pharmacophores and stores them in an internal data structure. """
    pharmacophore = []
    pml_pharmacophore = et.parse(path)
    if pml_pharmacophore.getroot().tag == 'pharmacophore':
        pml_pharmacophore = pml_pharmacophore.getroot()
    else:
        pharmacophore_number = len(pml_pharmacophore.findall('pharmacophore'))
        if pharmacophore_number > 0:
            if pharmacophore_number == 1:
                pml_pharmacophore = pml_pharmacophore.findall('pharmacophore')[0]
            else:
                user_prompt = ''
                while user_prompt not in ['yes', 'no']:
                    user_prompt = input('Pharmacophore file contains {} pharmacophores. Only one pharmacophore '
                                        'can be processed at a time. Do you want to continue with the first '
                                        'pharmacophore in the pml file? [yes/no]: '.format(pharmacophore_number))
                    if user_prompt == 'no':
                        sys.exit()
                    elif user_prompt == 'yes':
                        pml_pharmacophore = pml_pharmacophore.findall('pharmacophore')[0]
        else:
            update_user('Cannot find any pharmacophore in the pml file.', logger)
            sys.exit()
    for feature in pml_pharmacophore:
        pharmacophore.append(pml_feature(feature, pyrod_pharmacophore, logger))
    if pyrod_pharmacophore:
        pharmacophore = merge_pyrod_hydrogen_bonds(pharmacophore, logger)
    return pharmacophore


def pharmacophore_reader(path, pyrod_pharmacophore, logger):
    """ This function reads pharmacophores and translates them into the internal pharmacophore format.

    Format:
    ------------------------------------------------------------------------
    0   1 2             3   4                             5          6     7
    ------------------------------------------------------------------------
    0  hi M [0.0,0.0,0.0] 1.5                            []        0.0 150.1
    1  pi M [0.0,0.0,0.0] 1.5                            []        0.0  30.1
    2  ni M [0.0,0.0,0.0] 1.5                            []        0.0  30.1
    3  hd M [0.0,0.0,0.0] 1.5               [[3.0,0.0,0.0]]  1.9499999  30.1
    4  ha M [0.0,0.0,0.0] 1.5               [[3.0,0.0,0.0]]  1.9499999  30.1
    5 hd2 M [0.0,0.0,0.0] 1.5 [[3.0,0.0,0.0],[0.0,3.0,0.0]]  1.9499999  30.1
    6 ha2 M [0.0,0.0,0.0] 1.5 [[3.0,0.0,0.0],[0.0,3.0,0.0]]  1.9499999  30.1
    7 hda M [0.0,0.0,0.0] 1.5 [[3.0,0.0,0.0],[0.0,3.0,0.0]]  1.9499999  30.1
    8  ai M [0.0,0.0,0.0] 1.5               [[1.0,0.0,0.0]] 0.43633232  30.1
    ------------------------------------------------------------------------
    Legend:
    0 - index
    1 - type
    2 - flag (O - optional, M - mandatory)
    3 - core position
    4 - core tolerance
    5 - partner positions (hda feature with coordinates for first donor than acceptor)
    6 - partner tolerance
    7 - score
    """
    logger.info('Reading pharmacophore from {}.'.format(path))
    valid_formats = ['pml', 'pdb']
    pharmacophore_format = path.split('.')[-1]
    if pharmacophore_format not in valid_formats:
        update_user('Invalid pharmacophore format detected, only {} and {} are supported.'.format(
                    ', '.join(valid_formats[0:-1]), valid_formats[-1]), logger)
        sys.exit()
    pharmacophore = []
    if pharmacophore_format == 'pml':
        pharmacophore = pml_pharmacophore_reader(path, pyrod_pharmacophore, logger)
    elif pharmacophore_format == 'pdb':
        pharmacophore = pdb_pharmacophore_reader(path, pyrod_pharmacophore, logger)
    if not pyrod_pharmacophore:
        pharmacophore = renumber_features(pharmacophore)
    return pharmacophore
