""" PyRod - dynamic molecular interaction fields (dMIFs), based on tracing water molecules in MD simulations.

This module contains helper functions to write out data.
"""


# python standard libraries
import copy
import logging
import os
import pickle
import xml.etree.ElementTree as et

# external libraries
import numpy as np

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


def file_path(name, path):
    """ This function creates the path to a file. If the file already exists it will be deleted. """
    if not os.path.isdir(path):
        os.makedirs(path)
    if os.path.exists('/'.join([path, name])):
        os.remove('/'.join([path, name]))
    return


def pdb_text(positions, resname, scores, indices=None):
    """ This function generates a list of strings for writing pdb files. """
    template = '{0:<6s}{1:>5d} {2:>4s} {3:>3s} {4:>1s}{5:>4d}    {6:>8.3f}{7:>8.3f}{8:>8.3f}{9:>6.2f}{10:>6.2f}' \
               '          {11:>2s} \n'
    if indices is None:
        indices = range(1, len(positions) + 1)
    atoms = []
    for index, position, score in zip(indices, positions, scores):
        x, y, z = position[0], position[1], position[2]
        atoms.append(template.format('ATOM', index, 'X', resname, 'A', 1, x, y, z, 0, score, 'X'))
    return atoms


def cns_xplor_text(positions, scores, dmif_format):
    """ This function generates a list of strings for writing cns and xplor maps. """
    density = ['\n', '{0:>8}\n'.format(1), 'REMARKS CNS TEST\n']
    x_minimum, x_maximum, y_minimum, y_maximum, z_minimum, z_maximum, space = grid_parameters(positions)
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
    if 'cns' in dmif_format:
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


def pdb_writer(positions, name, path, resname='GRI', scores=None):
    """ This function writes out data as atoms to a pdb file. If scores are provided, each score is saved as b-factor
    in the respective atom. """
    if len(positions) > 99999:
        print('Decrease number of data points. Only 99999 data points (atoms) can be written to a pdb. You '
              'attempted to write {} data points.'.format(len(positions)))
        return
    if scores is None:
        scores = [0] * len(positions)
    if len(positions) != len(scores):
        print('Numbers of positions and scores must be equal. You provided {} positions and {} scores').format(
            len(positions), len(scores))
        return
    name = '.'.join([name, 'pdb'])
    file_path(name, path)
    with open('/'.join([path, name]), 'w') as pdb_file:
        atoms = pdb_text(positions, resname, scores)
        pdb_file.write(''.join(atoms))
    return


def dmif_writer(dmif, feature, file_format, name, directory):
    """ This function writes out dmifs as density maps. """
    valid_formats = ['xplor', 'cns', 'kont']
    if file_format not in valid_formats:
        print('Invalid dmif format, only {} and {} are supported.'.format(', '.join(valid_formats[:-1]),
                                                                          valid_formats[-1]))
        return
    positions = np.array([[x, y, z] for x, y, z in zip(dmif['x'], dmif['y'], dmif['z'])])
    scores = dmif[feature]
    name = '.'.join([name, file_format])
    file_path(name, directory)
    with open('/'.join([directory, name]), 'w') as dmif_file:
        if file_format in ['xplor', 'cns']:
            density = cns_xplor_text(positions, scores, file_format)
        else:
            density = kont_text(positions, scores)
        dmif_file.write(''.join(density))
    return


def pml_feature_point(pharmacophore, feature, weight):
    """ This function generates an xml branch for positive and negative ionizable features as well as hydrophobic
    interactions. """
    translate_features = {'hi': 'H', 'pi': 'PI', 'ni': 'NI'}
    point_name = translate_features[feature[1]]
    point_featureId = str(feature[0])
    point_optional = 'false'
    point_disabled = 'false'
    if weight:
        point_weight = str(feature[6])
    else:
        point_weight = '1.0'
    position_x3 = str(feature[2][0])
    position_y3 = str(feature[2][1])
    position_z3 = str(feature[2][2])
    position_tolerance = str(feature[3])
    point_attributes = {'name': point_name, 'featureId': point_featureId, 'optional': point_optional,
                        'disabled': point_disabled, 'weight': point_weight}
    point = et.SubElement(pharmacophore, 'point', attrib=point_attributes)
    position_attributes = {'x3': position_x3, 'y3': position_y3, 'z3': position_z3, 'tolerance': position_tolerance}
    et.SubElement(point, 'position', attrib=position_attributes)
    return


def pml_feature_vector(pharmacophore, feature, weight):
    """ This function generates an xml branch for hydrogen bonds. """
    translate_features = {'hd': 'HBD', 'hd2': 'HBD', 'ha': 'HBA', 'ha2': 'HBA', 'hda': ['HBD', 'HBA']}
    for index in range(len(feature[4])):
        vector_name = translate_features[feature[1]]
        vector_featureId = str(feature[0])
        vector_pointsToLigand = 'false'
        vector_hasSyntheticProjectedPoint = 'false'
        vector_optional = 'false'
        vector_disabled = 'false'
        vector_weight = '1.0'
        origin_x3 = str(feature[2][0])
        origin_y3 = str(feature[2][1])
        origin_z3 = str(feature[2][2])
        origin_tolerance = str(feature[3])
        target_x3 = str(feature[4][index][0])
        target_y3 = str(feature[4][index][1])
        target_z3 = str(feature[4][index][2])
        target_tolerance = '1.9499999'
        if feature[1] in ['ha', 'ha2']:
            vector_pointsToLigand = 'true'
            origin_x3 = str(feature[4][index][0])
            origin_y3 = str(feature[4][index][1])
            origin_z3 = str(feature[4][index][2])
            origin_tolerance = '1.9499999'
            target_x3 = str(feature[2][0])
            target_y3 = str(feature[2][1])
            target_z3 = str(feature[2][2])
            target_tolerance = str(feature[3])
        elif feature[1] == 'hda':
            vector_name = translate_features[feature[1]][index]
            if index == 1:
                vector_pointsToLigand = 'true'
                origin_x3 = str(feature[4][index][0])
                origin_y3 = str(feature[4][index][1])
                origin_z3 = str(feature[4][index][2])
                origin_tolerance = '1.9499999'
                target_x3 = str(feature[2][0])
                target_y3 = str(feature[2][1])
                target_z3 = str(feature[2][2])
                target_tolerance = str(feature[3])
        if len(feature[4]) == 2:
            vector_featureId = '_'.join([str(feature[0]), str(index + 1)])
        if weight:
            vector_weight = str(feature[6])
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


def pml_feature_volume(pharmacophore, feature):
    """ This function generates an xml branch for exclusion volumes. """
    translate_features = {'ev': 'exclusion'}
    volume_type = translate_features[feature[1]]
    volume_featureId = '_'.join(['ev', str(feature[0])])
    volume_optional = 'false'
    volume_disabled = 'false'
    volume_weight = '1.0'
    position_x3 = str(feature[2][0])
    position_y3 = str(feature[2][1])
    position_z3 = str(feature[2][2])
    position_tolerance = str(feature[3])
    volume_attributes = {'type': volume_type, 'featureId': volume_featureId, 'optional': volume_optional,
                         'disabled': volume_disabled, 'weight': volume_weight}
    volume = et.SubElement(pharmacophore, 'volume', attrib=volume_attributes)
    position_attributes = {'x3': position_x3, 'y3': position_y3, 'z3': position_z3, 'tolerance': position_tolerance}
    et.SubElement(volume, 'position', attrib=position_attributes)
    return


def pml_feature_plane(pharmacophore, feature, weight):
    """ This function generates an xml branch for aromatic interactions. """
    plane_name = 'AR'
    plane_featureId = str(feature[0])
    plane_optional = 'false'
    plane_disabled = 'false'
    if weight:
        plane_weight = str(feature[6])
    else:
        plane_weight = '1.0'
    position_x3 = str(feature[2][0])
    position_y3 = str(feature[2][1])
    position_z3 = str(feature[2][2])
    position_tolerance = str(feature[3])
    normal_x3 = str(feature[2][0] - feature[4][0][0])
    normal_y3 = str(feature[2][1] - feature[4][0][1])
    normal_z3 = str(feature[2][2] - feature[4][0][2])
    normal_tolerance = '0.43633232'
    plane_attributes = {'name': plane_name, 'featureId': plane_featureId, 'optional': plane_optional,
                        'disabled': plane_disabled, 'weight': plane_weight}
    plane = et.SubElement(pharmacophore, 'plane', attrib=plane_attributes)
    position_attributes = {'x3': position_x3, 'y3': position_y3, 'z3': position_z3, 'tolerance': position_tolerance}
    et.SubElement(plane, 'position', attrib=position_attributes)
    normal_attributes = {'x3': normal_x3, 'y3': normal_y3, 'z3': normal_z3, 'tolerance': normal_tolerance}
    et.SubElement(plane, 'normal', attrib=normal_attributes)
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
        maximal_score = 0
        for feature in features:
            if feature[1] in feature_class:
                if feature[6] > maximal_score:
                    maximal_score = feature[6]
        for feature in features:
            if feature[1] in feature_class:
                feature[6] = round(feature[6] / maximal_score, 2)
    return features


def pml_pharmacophore(features, name, weight):
    """ This function generates an xml tree that can be used to write pharmacophores to a pml file. """
    if weight:
        features = normalize_weights(copy.deepcopy(features))
    pharmacophore = et.Element('pharmacophore', attrib={'name': name, 'pharmacophoreType': 'LIGAND_SCOUT'})
    for feature_name in ['hi', 'pi', 'ni', 'hd', 'hd2', 'ha', 'ha2', 'hda', 'ai', 'ev']:
        for feature in features:
            if feature[1] == feature_name:
                pml_feature(pharmacophore, feature, weight)
    return et.ElementTree(pharmacophore)


def pdb_pharmacophore(features):
    """ This function generates list of strings that can be used to write pharmacophores to a pdb file. """
    pharmacophore = []
    for feature_name in feature_names:
        features_filtered = [_ for _ in features if _[1] == feature_name]
        if len(features_filtered) > 0:
            indices = []
            positions = []
            scores = []
            for feature in features_filtered:
                indices.append(feature[0])
                positions.append(feature[2])
                scores.append(feature[6])
            pharmacophore += pdb_text(positions, feature_name, scores, indices)
    return pharmacophore


def pharmacophore_writer(features, file_format, name, path, weight):
    """ This function writes out pharmacophores. """
    valid_formats = ['pml', 'pdb']
    if file_format not in valid_formats:
        print('Invalid pharmacophore format, only {} and {} are supported.'.format(valid_formats[0], valid_formats[1]))
        return
    name = '.'.join([name, file_format])
    file_path(name, path)
    if file_format == 'pdb':
        with open('/'.join([path, name]), 'w') as pharmacophore_file:
            pharmacophore = pdb_pharmacophore(features)
            pharmacophore_file.write(''.join(pharmacophore))
    elif file_format == 'pml':
        tree = pml_pharmacophore(features, name, weight)
        tree.write('/'.join([path, name]), encoding="UTF-8", xml_declaration=True)
    return


def pickle_writer(data, name, directory, logger):
    name = '.'.join([name, 'p'])
    update_user('Writing {} to {}.'.format(name, directory), logger)
    file_path(name, directory)
    pickle.dump(data, (open('/'.join([directory, name]), 'wb')))
    return


def setup_logger(name, directory, debugging):
    directory = '/'.join([directory, 'logs'])
    file_path('.'.join([name, 'log']), directory)
    handler = logging.FileHandler('/'.join([directory, '.'.join([name, 'log'])]))
    handler.setFormatter(logging.Formatter(fmt='%(asctime)s %(levelname)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
    logger = logging.getLogger(name)
    if debugging:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    return logger
