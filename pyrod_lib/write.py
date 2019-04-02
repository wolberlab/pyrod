""" PyRod - dynamic molecular interaction fields (dMIFs), based on tracing water molecules in MD simulations.

This module contains helper functions to write out data.
"""

# python standard libraries
import copy
import time
from itertools import starmap
import logging
import os
import pickle
import sys
import xml.etree.ElementTree as et

# external libraries
import numpy as np

# pyrod modules
try:
    from pyrod.pyrod_lib.lookup import feature_types
    from pyrod.pyrod_lib.math import mean, standard_deviation
    from pyrod.pyrod_lib.grid import grid_characteristics
except ImportError:
    from pyrod_lib.lookup import feature_types
    from pyrod_lib.math import mean, standard_deviation
    from pyrod_lib.grid import grid_characteristics


def file_path(name, directory):
    """ This function creates the path to a file. If the file already exists it will be deleted. """
    try:
        os.makedirs(directory)
    except FileExistsError:
        pass
    if os.path.exists('{}/{}'.format(directory, name)):
        os.remove('{}/{}'.format(directory, name))
    return


def pdb_line(atomid, atomname, resname, chain, resid, position, occupancy, b_factor, element):
    """ This function generates a strings for writing a line to a pdb file. """
    template = '{0:<6s}{1:>5d} {2:>4s} {3:>3s} {4:>1s}{5:>4d}    {6:>8.3f}{7:>8.3f}{8:>8.3f}{9:>6.2f}{10:>6.2f}' \
               '          {11:>2s}  \n'
    x, y, z = position[0], position[1], position[2]
    line = template.format('ATOM', atomid, atomname, resname, chain, resid, x, y, z, occupancy, b_factor, element)
    return line


def pdb_grid(positions, name, directory):
    """ This function writes out grid positions as atoms to a pdb file. """
    if len(positions) > 99999:
        print('Decrease number of data points. Only 99999 data points (atoms) can be written to a pdb. You '
              'attempted to write {} data points.'.format(len(positions)))
        sys.exit()
    name = '{}.pdb'.format(name)
    file_path(name, directory)
    with open('{}/{}'.format(directory, name), 'w') as pdb_file:
        lines = [pdb_line(counter + 1, 'X', 'GRI', 'A', counter + 1, position, 0, 0, 'X') for counter, position in
                 enumerate(positions)]
        pdb_file.write(''.join(lines))
    return


def pdb_pharmacophore(features, directory, name, weight):
    """ This function generates a list of strings describing a pharmacophore that is written to a pdb file.

    Format:
    ------------------------------------------------------------------------------
                   0   1 2   3           4       5       6     7     8
    ------------------------------------------------------------------------------
    ATOM      1    C  hi M   1      11.579  13.109  10.857  2.00164.07           X
    ATOM      2    C  pi M   2      11.579  13.109  10.857  1.50 30.32           X
    ATOM      3    C  ni M   3      11.579  13.109  10.857  1.50 29.57           X
    ATOM      4    C  ai M   4      11.579  13.109  10.857  1.50 40.30           X
    ATOM      5    P  ai M   4      11.579  14.109  10.857  0.44 40.30           X
    ATOM      6    C  hd M   5      11.579  13.109  10.857  1.50 50.07           X
    ATOM      7    P  hd M   5      11.579  15.709  10.857  1.95 50.07           X
    ATOM      8    C  ha M   6      11.579  13.109  10.857  1.50 12.07           X
    ATOM      9    P  ha M   6      14.179  13.109  10.857  1.95 12.07           X
    ATOM     10    C hd2 O   7      11.579  13.109  10.857  1.50 23.07           X
    ATOM     11    P hd2 O   7      11.579  15.709  10.857  1.95 23.07           X
    ATOM     12    P hd2 O   7      14.179  13.109  10.857  1.95 23.07           X
    ATOM     13    C ha2 M   8      11.579  13.109  10.857  1.50 37.07           X
    ATOM     14    P ha2 M   8      11.579  15.709  10.857  1.95 37.07           X
    ATOM     15    P ha2 M   8      14.179  13.109  10.857  1.95 37.07           X
    ATOM     16    C hda M   9       3.500 -22.500  -8.500  1.50 39.00           X
    ATOM     17   Pd hda M   9       6.100 -22.500  -8.500  1.95 39.00           X
    ATOM     18   Pa hda M   9       3.500 -19.900  -8.500  1.95 39.00           X
    ATOM     19    C  ev M  10      11.579  13.109  10.857  1.00  0.00           X
    END
    ------------------------------------------------------------------------------
    Legend:
    0 - part (C - core, P - partner, Pd - donor partner, Pa - acceptor partner)
    1 - type
    2 - flag (O - optional, M - mandatory)
    3 - id
    4 - x coordinate
    5 - y coordinate
    6 - z coordinate
    7 - tolerance
    8 - score
    """
    pharmacophore = []
    atomid = 1
    for feature in features:
        feature_weight = 1.0
        if weight:
            feature_weight = feature[7]
        pharmacophore.append(pdb_line(atomid, 'C', feature[1], feature[2], feature[0], feature[3], feature[4],
                                      feature_weight, 'X'))
        atomid += 1
        if feature[1] in ['ha', 'hd', 'ha2', 'hd2', 'hda', 'ai']:
            part = 'P'
            if feature[1] == 'hda':
                part = 'Pd'
            pharmacophore.append(pdb_line(atomid, part, feature[1], feature[2], feature[0], feature[5][0], feature[6],
                                          feature_weight, 'X'))
            atomid += 1
            if feature[1] in ['ha2', 'hd2', 'hda']:
                if feature[1] == 'hda':
                    part = 'Pa'
                pharmacophore.append(pdb_line(atomid, part, feature[1], feature[2], feature[0], feature[5][1],
                                              feature[6], feature_weight, 'X'))
                atomid += 1
    with open('{}/{}'.format(directory, name), 'w') as pharmacophore_file:
        pharmacophore_file.write(''.join(pharmacophore))
    return


def pml_feature_point(pharmacophore, feature, weight):
    """ This function generates an xml branch for positive and negative ionizable features as well as hydrophobic
    interactions. """
    translate_features = {'hi': 'H', 'pi': 'PI', 'ni': 'NI'}
    point_name = translate_features[feature[1]]
    point_featureId = '{}_{}'.format(feature[1], feature[0])
    point_optional = 'false'
    point_disabled = 'false'
    point_weight = '1.0'
    if weight:
        point_weight = str(feature[7])
    position_x3, position_y3, position_z3 = str(feature[3][0]), str(feature[3][1]), str(feature[3][2])
    position_tolerance = str(feature[4])
    point_attributes = {'name': point_name, 'featureId': point_featureId, 'optional': point_optional,
                        'disabled': point_disabled, 'weight': point_weight}
    point = et.SubElement(pharmacophore, 'point', attrib=point_attributes)
    position_attributes = {'x3': position_x3, 'y3': position_y3, 'z3': position_z3, 'tolerance': position_tolerance}
    et.SubElement(point, 'position', attrib=position_attributes)
    return


def pml_feature_vector(pharmacophore, feature, weight):
    """ This function generates an xml branch for hydrogen bonds. """
    for index in range(len(feature[5])):  # all as donor
        vector_name = 'HBD'
        vector_featureId = '{}_{}'.format(feature[1], feature[0])
        if feature[1] in ['ha2', 'hd2', 'hda']:
            vector_featureId = '{}_{}_{}'.format(feature[1], feature[0], index)
        vector_pointsToLigand = 'false'
        vector_hasSyntheticProjectedPoint = 'false'
        vector_optional = 'false'
        vector_disabled = 'false'
        vector_weight = '1.0'
        if weight:
            vector_weight = str(feature[7])
        origin_x3, origin_y3, origin_z3 = str(feature[3][0]), str(feature[3][1]), str(feature[3][2])
        origin_tolerance = str(feature[4])
        target_x3, target_y3, target_z3 = [str(feature[5][index][0]), str(feature[5][index][1]),
                                           str(feature[5][index][2])]
        target_tolerance = str(feature[6])
        if feature[1] in ['ha', 'ha2'] or (feature[1] == 'hda' and index == 1):  # switch to acceptor
            vector_name = 'HBA'
            vector_pointsToLigand = 'true'
            origin_x3, origin_y3, origin_z3, target_x3, target_y3, target_z3 = [target_x3, target_y3, target_z3,
                                                                                origin_x3, origin_y3, origin_z3]
            origin_tolerance, target_tolerance = target_tolerance, origin_tolerance
        vector_attributes = {'name': vector_name, 'featureId': vector_featureId,
                             'pointsToLigand': vector_pointsToLigand,
                             'hasSyntheticProjectedPoint': vector_hasSyntheticProjectedPoint,
                             'optional': vector_optional,
                             'disabled': vector_disabled, 'weight': vector_weight}
        vector = et.SubElement(pharmacophore, 'vector', attrib=vector_attributes)
        origin_attributes = {'x3': origin_x3, 'y3': origin_y3, 'z3': origin_z3, 'tolerance': origin_tolerance}
        et.SubElement(vector, 'origin', attrib=origin_attributes)
        target_attributes = {'x3': target_x3, 'y3': target_y3, 'z3': target_z3, 'tolerance': target_tolerance}
        et.SubElement(vector, 'target', attrib=target_attributes)
    return


def pml_feature_plane(pharmacophore, feature, weight):
    """ This function generates an xml branch for aromatic interactions. """
    plane_name = 'AR'
    plane_featureId = '{}_{}'.format('ai', feature[0])
    plane_optional = 'false'
    plane_disabled = 'false'
    plane_weight = '1.0'
    if weight:
        plane_weight = str(feature[7])
    position_x3, position_y3, position_z3 = str(feature[3][0]), str(feature[3][1]), str(feature[3][2])
    position_tolerance = str(feature[4])
    normal_x3, normal_y3, normal_z3 = [str(feature[3][0] - feature[5][0][0]), str(feature[3][1] - feature[5][0][1]),
                                       str(feature[3][2] - feature[5][0][2])]
    normal_tolerance = str(feature[6])
    plane_attributes = {'name': plane_name, 'featureId': plane_featureId, 'optional': plane_optional,
                        'disabled': plane_disabled, 'weight': plane_weight}
    plane = et.SubElement(pharmacophore, 'plane', attrib=plane_attributes)
    position_attributes = {'x3': position_x3, 'y3': position_y3, 'z3': position_z3, 'tolerance': position_tolerance}
    et.SubElement(plane, 'position', attrib=position_attributes)
    normal_attributes = {'x3': normal_x3, 'y3': normal_y3, 'z3': normal_z3, 'tolerance': normal_tolerance}
    et.SubElement(plane, 'normal', attrib=normal_attributes)
    return


def pml_feature_volume(pharmacophore, feature):
    """ This function generates an xml branch for exclusion volumes. """
    volume_type = 'exclusion'
    volume_featureId = '{}_{}'.format('ev', feature[0])
    volume_optional = 'false'
    volume_disabled = 'false'
    volume_weight = '1.0'
    position_x3, position_y3, position_z3 = str(feature[3][0]), str(feature[3][1]), str(feature[3][2])
    position_tolerance = str(feature[4])
    volume_attributes = {'type': volume_type, 'featureId': volume_featureId, 'optional': volume_optional,
                         'disabled': volume_disabled, 'weight': volume_weight}
    volume = et.SubElement(pharmacophore, 'volume', attrib=volume_attributes)
    position_attributes = {'x3': position_x3, 'y3': position_y3, 'z3': position_z3, 'tolerance': position_tolerance}
    et.SubElement(volume, 'position', attrib=position_attributes)
    return


def pml_feature(pharmacophore, feature, weight):
    """ This function distributes features according to their feature type to the appropriate feature function. """
    if feature[1] in ['hi', 'ni', 'pi']:
        pml_feature_point(pharmacophore, feature, weight)
    elif feature[1] in ['hd', 'hd2', 'ha', 'ha2', 'hda']:
        pml_feature_vector(pharmacophore, feature, weight)
    elif feature[1] == 'ai':
        pml_feature_plane(pharmacophore, feature, weight)
    elif feature[1] == 'ev':
        pml_feature_volume(pharmacophore, feature)
    return


def normalize_weights(features):
    """ This function normalizes the scores for each feature class. """
    for feature_class in [['hi'], ['pi', 'ni'], ['hd', 'hd2', 'ha', 'ha2', 'hda'], ['ai']]:
        scores = [feature[7] for feature in features if feature[1] in feature_class]
        if len(scores) > 0:
            maximal_score = max(scores)
            features = [feature[:7] + [feature[7] / maximal_score] if feature[1] in feature_class
                        else feature for feature in features]
    return features


def indent_xml(element, level=0):
    """ This function adds indentation to an xml structure for pretty printing. """
    i = "\n" + level*"  "
    if len(element):
        if not element.text or not element.text.strip():
            element.text = i + "  "
        if not element.tail or not element.tail.strip():
            element.tail = i
        for element in element:
            indent_xml(element, level + 1)
        if not element.tail or not element.tail.strip():
            element.tail = i
    else:
        if level and (not element.tail or not element.tail.strip()):
            element.tail = i


def pml_pharmacophore(features, directory, name, weight):
    """ This function generates an xml tree describing a pharmacophore that is written to a pml file. """
    pharmacophore = et.Element('pharmacophore', attrib={'name': name, 'pharmacophoreType': 'LIGAND_SCOUT'})
    for feature in features:
        pml_feature(pharmacophore, feature, weight)
    indent_xml(pharmacophore)
    tree = et.ElementTree(pharmacophore)
    tree.write('{}/{}'.format(directory, name), encoding="UTF-8", xml_declaration=True)
    return


def pharmacophore_writer(features, file_formats, name, directory, weight, logger):
    """ This function writes out pharmacophores. """
    valid_formats = ['pml', 'pdb']
    if not all(elem in valid_formats for elem in file_formats):
        update_user('Invalid pharmacophore format detected, only {} and {} are supported.'.format(
                    ', '.join(valid_formats[0:-1]), valid_formats[-1]), logger)
        sys.exit()
    file_path(name, directory)
    if weight:
        features = normalize_weights(features)
    if 'pdb' in file_formats:
        pdb_pharmacophore(features, directory, '{}.{}'.format(name, 'pdb'), weight)
    if 'pml' in file_formats:
        pml_pharmacophore(features, directory, '{}.{}'.format(name, 'pml'), weight)
    return


def cns_xplor_text(positions, scores, dmif_format):
    """ This function generates a list of strings for writing cns and xplor maps. """
    density = ['\n', '{0:>8}\n'.format(1), 'REMARKS {} file generated by PyRod\n'.format(dmif_format)]
    x_minimum, x_maximum, y_minimum, y_maximum, z_minimum, z_maximum, space = grid_characteristics(positions)
    density.append('{0:>8}{1:>8}{2:>8}{3:>8}{4:>8}{5:>8}{6:>8}{7:>8}{8:>8}\n'.format(
        int(((x_maximum - x_minimum) / space) + 1), int(x_minimum / space), int(x_maximum / space),
        int(((y_maximum - y_minimum) / space) + 1), int(y_minimum / space), int(y_maximum / space),
        int(((z_maximum - z_minimum) / space) + 1), int(z_minimum / space), int(z_maximum / space)))
    density.append('{0:>12.5E}{1:>12.5E}{2:>12.5E}{3:>12.5E}{3:>12.5E}{3:>12.5E}\n'.format(
        x_maximum - x_minimum, y_maximum - y_minimum, z_maximum - z_minimum, 90))
    density.append('ZYX\n')
    index_z = 0
    index_xy = 0
    xy = int(((x_maximum - x_minimum) / space) + 1) * int(((y_maximum - y_minimum) / space) + 1)
    for score in scores:
        if index_xy % xy == 0:
            if index_xy % 6 == 0:
                density.append('{0:>8}\n'.format(index_z))
            else:
                density.append('\n{0:>8}\n'.format(index_z))
            index_z += 1
            index_xy = 0
        density.append('{0:>12.5E}'.format(score))
        index_xy += 1
        if index_xy % 6 == 0:
            density.append('\n')
    if dmif_format == 'cns':
        if index_xy % 6 == 0:
            density.append('{0:>8}\n'.format(-9999))
        else:
            density.append('\n{0:>8}\n'.format(-9999))
        density.append(' {0:>12.5E}{1:>12.5E}\n'.format(mean(scores), standard_deviation(scores)))
        density.append('\n')
    return density


def kont_text(positions, scores):
    """ This function generates a list of strings for writing kont maps. """
    density = []
    for index, position in enumerate(positions):
        x, y, z = position[0], position[1], position[2]
        density.append('{0:>7d}  {1:>8.3f}{2:>8.3f}{3:>8.3f}\n'.format(index + 1, x, y, z))
    density.append(' {0:>78s}'.format('separation line'))
    for score in scores:
        density.append('\n{0:>8.3f}'.format(score))
    return density


def dmif_writer(scores, positions, file_format, name, directory, logger):
    """ This function writes out dmifs as density maps. """
    valid_formats = ['xplor', 'cns', 'kont']
    if file_format not in valid_formats:
        update_user('Invalid dmif format, only {} and {} are supported.'.format(', '.join(valid_formats[:-1]),
                    valid_formats[-1]), logger)
        return
    name = '{}.{}'.format(name, file_format)
    file_path(name, directory)
    with open('{}/{}'.format(directory, name), 'w') as dmif_file:
        if file_format in ['xplor', 'cns']:
            density = cns_xplor_text(positions, scores, file_format)
        else:
            density = kont_text(positions, scores)
        dmif_file.write(''.join(density))
    return


def pickle_writer(data, name, directory, logger):
    """ This function writes data using the pickle module. """
    name = '{}.pkl'.format(name)
    update_user('Writing {} to {}.'.format(name, directory), logger)
    file_path(name, directory)
    with open('{}/{}'.format(directory, name), 'wb') as file:
        pickle.dump(data, file, pickle.HIGHEST_PROTOCOL)
    return


def setup_logger(name, directory, debugging):
    """ This function setups and returns a logger for writing to log files. """
    directory = '{}/logs'.format(directory)
    file_path('{}.log'.format(name), directory)
    handler = logging.FileHandler('{}/{}.log'.format(directory, name))
    handler.setFormatter(logging.Formatter(fmt='%(asctime)s %(levelname)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
    logger = logging.getLogger(name)
    if debugging:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    return logger


def time_to_text(seconds):
    """ This function converts a time in seconds into a reasonable format. """
    if seconds > 60:
        if seconds > 3600:
            if seconds > 86400:
                if seconds > 1209600:
                    if seconds > 31449600:
                        time_as_text = 'years'
                    else:
                        time_as_text = '{} weeks'.format(round(seconds / 1209600, 1))
                else:
                    time_as_text = '{} d'.format(round(seconds / 86400, 1))
            else:
                time_as_text = '{} h'.format(round(seconds / 3600, 1))
        else:
            time_as_text = '{} min'.format(round(seconds / 60, 1))
    else:
        time_as_text = '{} s'.format(int(seconds))
    return time_as_text


def bytes_to_text(bytes_number):
    """ This function converts a size of an object in bytes into a reasonable format. """
    if bytes_number > 1000:
        if bytes_number > 1000000:
            if bytes_number > 1000000000:
                if bytes_number > 1000000000000:
                    bytes_as_text = '{} TB'.format(round(bytes_number / 1000000000000, 1))
                else:
                    bytes_as_text = '{} GB'.format(round(bytes_number / 1000000000, 1))
            else:
                bytes_as_text = '{} MB'.format(round(bytes_number / 1000000))
        else:
            bytes_as_text = '{} KB'.format(round(bytes_number / 1000))
    else:
        bytes_as_text = '{} B'.format(bytes_number)
    return bytes_as_text


def update_progress(progress, progress_info, eta):
    """ This function writes a progress bar to the terminal. """
    bar_length = 10
    block = int(bar_length * progress)
    if progress == 1.0:
        status = '         Done\n'
    else:
        status = '  ETA {:8}'.format(time_to_text(eta))
    text = '\r{}: [{}] {:>5.1f}%{}'.format(progress_info, '=' * block + ' ' * (bar_length - block), progress * 100,
                                           status)
    sys.stdout.write(text)
    sys.stdout.flush()
    return


def update_user(text, logger):
    """ This function writes information to the terminal and to a log file. """
    print(text)
    logger.info(text)
    return
