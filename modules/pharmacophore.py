""" PyRod - dynamic molecular interaction fields (dMIFs), based on tracing water molecules in MD simulations.

This module contains functions to generate features and pharmacophores.
"""

# python standard libraries
import copy
from itertools import combinations
import os
import sys
import time
import xml.etree.ElementTree as et

# external libraries
import numpy as np
from scipy.spatial import cKDTree

# pyrod modules
try:
    from pyrod.modules.helper_math import distance
    from pyrod.modules.helper_pharmacophore import center, feature_tolerance, maximal_feature_tolerance, \
        maximal_sum_of_scores, generate_feature, evaluate_pharmacophore
    from pyrod.modules.helper_read import pickle_reader
    from pyrod.modules.helper_update import update_progress, update_user, bytes_to_text
    from pyrod.modules.helper_write import file_path, pml_feature_volume, setup_logger
    from pyrod.modules.lookup import grid_list_dict
except ImportError:
    from modules.helper_math import distance
    from modules.helper_pharmacophore import center, feature_tolerance, maximal_feature_tolerance, \
        maximal_sum_of_scores, generate_feature, evaluate_pharmacophore
    from modules.helper_read import pickle_reader
    from modules.helper_update import update_progress, update_user, bytes_to_text
    from modules.helper_write import file_path, pml_feature_volume, setup_logger
    from modules.lookup import grid_list_dict


def exclusion_volume_generator(dmif, directory, debugging, shape_cutoff, restrictive):
    logger = setup_logger('exclusion_volumes', directory, debugging)
    update_user('Generating exclusion volumes.', logger)
    grid_space = 0.5
    exclusion_volume_space = 4
    if restrictive:
        exclusion_volume_space = 2
    grid_tree = cKDTree([[x, y, z] for x, y, z in zip(dmif['x'], dmif['y'], dmif['z'])])
    dtype = [('x', float), ('y', float), ('z', float), ('shape', int), ('count', int)]
    dmif_shape = np.array([(x, y, z, shape, 0) for x, y, z, shape in zip(dmif['x'], dmif['y'], dmif['z'], dmif['shape'])
                           if shape < shape_cutoff], dtype=dtype)
    positions = np.array([[x, y, z] for x, y, z in zip(dmif_shape['x'], dmif_shape['y'], dmif_shape['z'])])
    shape_tree = cKDTree(positions)
    shape_grid_size = len(dmif_shape)
    # store number of neighbors with shape score smaller than shape_cutoff for grid points
    for index in range(shape_grid_size):
        dmif_shape['count'][index] = len(shape_tree.query_ball_point(positions[index], grid_space * 4))
    # sort for neighbor count
    dmif_shape = np.sort(dmif_shape, order='count')
    # rebuild positions and shape_tree
    positions = np.array([[x, y, z] for x, y, z in zip(dmif_shape['x'], dmif_shape['y'], dmif_shape['z'])])
    shape_tree = cKDTree(positions)
    used = []
    exclusion_volumes = []
    counter = 1
    start = time.time()
    for index in range(shape_grid_size):
        # grid_point index should not be in used list
        if index not in used:
            neighbor_list = shape_tree.query_ball_point(positions[index], exclusion_volume_space / 2)
            # elements of neighbor_list should not be in used list
            if len(set(neighbor_list + used)) == len(neighbor_list) + len(used):
                # grid_point should not be at border of grid
                if len(grid_tree.query_ball_point(positions[index], r=grid_space * 2)) == 33:
                    # grid_point should not be directly at border of binding pocket
                    if len(shape_tree.query_ball_point(positions[index], r=grid_space)) == 7:
                        # grid_point should not be surrounded by grid_points outside the binding pocket
                        if len(shape_tree.query_ball_point(positions[index], r=grid_space * 2)) < 33:
                            exclusion_volumes.append([counter, 'ev', positions[index], 1, [], 0, 0])
                            counter += 1
                            used += neighbor_list
        eta = ((time.time() - start) / (index + 1)) * (shape_grid_size - (index + 1))
        update_progress(float(index + 1) / shape_grid_size, 'Progress of exclusion volume generation', eta)
        logger.debug('Passed grid index {}.'.format(index))
    update_user('Finished with generation of {} exclusion volumes.'.format(len(exclusion_volumes)), logger)
    return exclusion_volumes


def features_generator(positions, feature_scores, feature_name, features_per_feature_type, directory, partner_path,
                       debugging):
    """ This function generates features with variable tolerance based on a global maximum search algorithm. """
    logger = setup_logger('_'.join(['features', feature_name]), directory, debugging)
    update_user('Starting {} feature generation.'.format(feature_name), logger)
    if partner_path is None:
        partner_path = directory
    if feature_name in grid_list_dict.keys():
        partners = pickle_reader(partner_path + '/data/' + feature_name + '.pkl', feature_name + '.pkl', logger)
    else:
        partners = []
    score_minimum = 1
    tree = cKDTree(positions)
    generated_features = []
    not_used = range(len(feature_scores))
    used = []
    while feature_scores[not_used].max() >= score_minimum:
        feature_maximum = feature_scores[not_used].max()
        logger.debug('Feature {} maximum of remaining grid points at {}.'.format(feature_name, feature_maximum))
        indices_not_checked = np.where(abs(feature_scores - feature_maximum) < 1e-8)[0]
        indices = []
        # check if grid points within minimum tolerance already used for features
        for index_not_checked in indices_not_checked:
            feature_indices = tree.query_ball_point(positions[index_not_checked], r=1.5)
            if len(feature_indices) + len(used) == len(set(feature_indices + used)):
                indices.append(index_not_checked)
            else:
                not_used = [x for x in not_used if x != index_not_checked]
        if len(indices) > 0:
            # check if only one grid point
            if len(indices) == 1:
                index = indices[0]
                tolerance, feature_indices = feature_tolerance(positions[index], tree, feature_scores, feature_maximum)
            # if more than one grid point, search for the ones with the biggest tolerance
            else:
                tolerance, indices_maximal_tolerance, feature_indices_list = \
                    maximal_feature_tolerance(indices, positions, tree, feature_scores, feature_maximum)
                # if more than one grid point with biggest tolerance, search for the one with the biggest score
                if len(indices_maximal_tolerance) > 1:
                    index, feature_indices = maximal_sum_of_scores(feature_scores, indices_maximal_tolerance,
                                                                   feature_indices_list)
                else:
                    index = indices_maximal_tolerance[0]
                    feature_indices = feature_indices_list[0]
            if len(feature_indices) + len(used) > len(set(feature_indices + used)):
                not_used = [x for x in not_used if x != index]
            else:
                generated_features.append(generate_feature(feature_name, index, positions, partners, feature_scores,
                                                           tolerance))
                not_used = [x for x in not_used if x not in feature_indices]
                used += feature_indices
            if len([x for x in generated_features if x[0] == feature_name]) >= features_per_feature_type:
                break
    update_user('Generated {} {} features.'.format(len([x for x in generated_features if x[0] == feature_name]),
                                                   feature_name), logger)
    return generated_features


def library_generator(pharmacophore_path, library_dict, library_path, make_mandatory, pyrod_pharmacophore):
    super_pharmacophore, pharmacophore_library = [], []
    essential_hb_features, essential_hi_features, essential_ai_features, essential_ii_features = [], [], [], []
    optional_hb_features, optional_hi_features, optional_ai_features, optional_ii_features = [], [], [], []
    exclusion_volumes = []
    tree = et.parse(pharmacophore_path)
    if tree.getroot().tag == 'pharmacophore':
        super_pharmacophore = tree.getroot()
    else:
        pharmacophore_number = len(tree.findall('pharmacophore'))
        if pharmacophore_number > 0:
            if pharmacophore_number == 1:
                super_pharmacophore = tree.findall('pharmacophore')[0]
            else:
                user_prompt = ''
                while user_prompt not in ['yes', 'no']:
                    user_prompt = input('Pharmacophore file contains {} pharmacophores. Only one pharmacophore '
                                        'can be processed at a time. Do you want to continue with the first '
                                        'pharmacophore in the pml file? [yes/no]: '.format(pharmacophore_number))
                    if user_prompt == 'no':
                        sys.exit()
                    elif user_prompt == 'yes':
                        super_pharmacophore = tree.findall('pharmacophore')[0]
        else:
            print('Cannot find any pharmacophore in the pml file.')
            sys.exit()
    # analyzing pharmacophore
    for index, feature in enumerate(super_pharmacophore):
        if feature.tag == 'volume':
            exclusion_volumes.append(index)
        else:
            if feature.attrib['name'] in ['HBD', 'HBA']:
                if feature.attrib['optional'] == 'true':
                    optional_hb_features.append(index)
                else:
                    essential_hb_features.append(index)
            elif feature.attrib['name'] == 'H':
                if feature.attrib['optional'] == 'true':
                    optional_hi_features.append(index)
                else:
                    essential_hi_features.append(index)
            elif feature.attrib['name'] in ['PI', 'NI']:
                if feature.attrib['optional'] == 'true':
                    optional_ii_features.append(index)
                else:
                    essential_ii_features.append(index)
            elif feature.attrib['name'] == 'AR':
                if feature.attrib['optional'] == 'true':
                    optional_ai_features.append(index)
                else:
                    essential_ai_features.append(index)
    # generating pharmacophore library
    for hbs in [combinations(optional_hb_features, x) for x in
                range(library_dict['minimal hydrogen bonds'] - len(essential_hb_features),
                      library_dict['maximal hydrogen bonds'] - len(essential_hb_features) + 1)]:
        for hbs_ in hbs:
            for his in [combinations(optional_hi_features, x) for x in
                        range(library_dict['minimal hydrophobic interactions'] - len(essential_hi_features),
                              library_dict['maximal hydrophobic interactions'] - len(essential_hi_features) + 1)]:
                for his_ in his:
                    for ais in [combinations(optional_ai_features, x) for x in
                                range(library_dict['minimal aromatic interactions'] - len(essential_ai_features),
                                      library_dict['maximal aromatic interactions'] - len(essential_ai_features) + 1)]:
                        for ais_ in ais:
                            for iis in [combinations(optional_ii_features, x) for x in
                                        range(library_dict['minimal ionizable interactions'] -
                                              len(essential_ii_features),
                                              library_dict['maximal ionizable interactions'] -
                                              len(essential_ii_features) + 1)]:
                                for iis_ in iis:
                                    pharmacophore = (essential_hb_features + list(hbs_) +
                                                     essential_hi_features + list(his_) +
                                                     essential_ai_features + list(ais_) +
                                                     essential_ii_features + list(iis_))
                                    if evaluate_pharmacophore(pharmacophore, super_pharmacophore,
                                                              library_dict['minimal features'],
                                                              library_dict['maximal features'], pyrod_pharmacophore):
                                        pharmacophore_library.append(pharmacophore)
    feature_indices = (essential_hb_features + essential_hi_features + essential_ii_features + essential_ai_features +
                       optional_hb_features + optional_hi_features + optional_ii_features + optional_ai_features)
    # generate pharmacophore template for pharmacophore writing
    pharmacophore_template = et.Element('pharmacophore', attrib={'name': 'template',
                                        'pharmacophoreType': 'LIGAND_SCOUT'})
    for feature in super_pharmacophore:
        if feature.tag == 'point':
            point = et.SubElement(pharmacophore_template, feature.tag, feature.attrib)
            et.SubElement(point, feature.find('position').tag, feature.find('position').attrib)
        elif feature.tag == 'vector':
            vector = et.SubElement(pharmacophore_template, feature.tag, feature.attrib)
            et.SubElement(vector, feature.find('origin').tag, feature.find('origin').attrib)
            et.SubElement(vector, feature.find('target').tag, feature.find('target').attrib)
        elif feature.tag == 'volume':
            volume = et.SubElement(pharmacophore_template, feature.tag, feature.attrib)
            et.SubElement(volume, feature.find('position').tag, feature.find('position').attrib)
        elif feature.tag == 'plane':
            plane = et.SubElement(pharmacophore_template, feature.tag, feature.attrib)
            et.SubElement(plane, feature.find('position').tag, feature.find('position').attrib)
            et.SubElement(plane, feature.find('normal').tag, feature.find('normal').attrib)
    if make_mandatory:
        for feature in pharmacophore_template.findall('./*[@optional="true"]'):
            feature.attrib['optional'] = "false"
    file_path('/'.join([library_path, 'super_pharmacophore.pml']), library_path)
    et.ElementTree(pharmacophore_template).write('/'.join([library_path, 'super_pharmacophore.pml']), encoding='UTF-8',
                                                 xml_declaration=True)
    pharmacophore_library_size = bytes_to_text(os.path.getsize('/'.join([library_path, 'super_pharmacophore.pml'])) *
                                               len(pharmacophore_library))
    # ask user if number and space of pharmacophores is okay
    user_prompt = ''
    while user_prompt not in ['yes', 'no']:
        user_prompt = input('{} pharmacophores will be written taking about {} of space.\nDo you want to continue? '
                            '[yes/no]: '.format(len(pharmacophore_library), pharmacophore_library_size))
        if user_prompt == 'no':
            sys.exit()
    start = time.time()
    # write pharmacophores
    for counter, pharmacophore_entry in enumerate(pharmacophore_library):
        additional_exclusion_volumes = 0
        name = '.'.join([str(counter), 'pml'])
        pharmacophore = copy.deepcopy(pharmacophore_template)
        pharmacophore.attrib['name'] = name
        forbidden_featureId_list = []
        # collect featureIds of forbidden features
        for index in feature_indices:
            if index not in pharmacophore_entry:
                forbidden_featureId_list.append(pharmacophore[index].attrib['featureId'])
        # remove forbidden features
        for forbidden_featureId in forbidden_featureId_list:
            pharmacophore.remove(pharmacophore.find('./*[@featureId="{}"]'.format(forbidden_featureId)))
        # add exclusion volumes for hydrogen bond partners
        for feature in pharmacophore.findall('vector'):
            additional_exclusion_volumes += 1
            position = 'target'
            if feature.attrib['name'] == 'HBA':
                position = 'origin'
            position = [float(feature.find(position).attrib['x3']),
                        float(feature.find(position).attrib['y3']),
                        float(feature.find(position).attrib['z3'])]
            pml_feature_volume(pharmacophore, [len(exclusion_volumes) + additional_exclusion_volumes, 'ev',
                                               position, 1.0])
        file_path(name, library_path)
        et.ElementTree(pharmacophore).write('/'.join([library_path, name]), encoding='UTF-8', xml_declaration=True)
        update_progress((counter + 1) / len(pharmacophore_library),
                        'Writing {} pharmacophores'.format(len(pharmacophore_library)),
                        ((time.time() - start) / (counter + 1)) * (len(pharmacophore_library) - (counter + 1)))
    return
