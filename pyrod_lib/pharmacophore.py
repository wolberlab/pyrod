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
    from pyrod.pyrod_lib.lookup import grid_list_dict
    from pyrod.pyrod_lib.math import distance
    from pyrod.pyrod_lib.pharmacophore_helper import get_center, get_core_tolerance, get_maximal_core_tolerance, \
        get_maximal_sum_of_scores, get_partner_positions, get_partner_tolerance, evaluate_pharmacophore
    from pyrod.pyrod_lib.read import pickle_reader, pharmacophore_reader
    from pyrod.pyrod_lib.write import file_path, setup_logger, pharmacophore_writer, update_progress, update_user, \
        bytes_to_text
except ImportError:
    from pyrod_lib.lookup import grid_list_dict
    from pyrod_lib.math import distance
    from pyrod_lib.pharmacophore_helper import get_center, get_core_tolerance, get_maximal_core_tolerance, \
        get_maximal_sum_of_scores, get_partner_positions, get_partner_tolerance, evaluate_pharmacophore
    from pyrod_lib.read import pickle_reader, pharmacophore_reader
    from pyrod_lib.write import file_path, setup_logger, pharmacophore_writer, bytes_to_text, update_progress, \
        update_user


def generate_exclusion_volumes(dmif, directory, debugging, shape_cutoff, restrictive):
    """ This function generates exclusion volumes. The exclusion volumes are described with a list of properties as
    follows.

    Format:
    ------------------------------------------------------------------------
    0   1 2             3   4                             5          6     7
    ------------------------------------------------------------------------
    0  ev M [0.0,0.0,0.0] 1.0                            []        0.0   0.0
    1  ev M [2.0,0.0,0.0] 1.0                            []        0.0   0.0
    ------------------------------------------------------------------------
    Legend:
    0 - index
    1 - type
    2 - flag (O - optional, M - mandatory)
    3 - core position
    4 - core tolerance [not needed for exclusion volumes]
    5 - partner positions [not needed for exclusion volumes]
    6 - partner tolerance [not needed for exclusion volumes]
    7 - score [not needed for exclusion volumes]
    """
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
                            exclusion_volumes.append([counter, 'ev', 'M', positions[index], 1.0, [], 0.0, 0.0])
                            counter += 1
                            used += neighbor_list
        eta = ((time.time() - start) / (index + 1)) * (shape_grid_size - (index + 1))
        update_progress(float(index + 1) / shape_grid_size, 'Progress of exclusion volume generation', eta)
        logger.debug('Passed grid index {}.'.format(index))
    update_user('Finished with generation of {} exclusion volumes.'.format(len(exclusion_volumes)), logger)
    return exclusion_volumes


def generate_features(positions, feature_scores, feature_type, features_per_feature_type, directory, partner_path,
                      debugging, total_number_of_features, start, feature_counter, results):
    """ This function generates features with variable tolerance based on a global maximum search algorithm. The
    features are described with a list of properties as follows.

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
    logger = setup_logger('_'.join(['features', feature_type]), directory, debugging)
    #update_user('Starting {} feature generation.'.format(feature_type), logger)
    if partner_path is None:
        partner_path = directory + '/data'
    if feature_type in grid_list_dict.keys():
        partners = pickle_reader(partner_path + '/' + feature_type + '.pkl', feature_type + '.pkl', logger)
    else:
        partners = [[]] * len(positions)
    score_minimum = 1
    tree = cKDTree(positions)
    generated_features = []
    not_used = range(len(feature_scores))
    used = []
    while feature_scores[not_used].max() >= score_minimum:
        feature_maximum = feature_scores[not_used].max()
        logger.debug('Feature {} maximum of remaining grid points at {}.'.format(feature_type, feature_maximum))
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
                core_tolerance, feature_indices = get_core_tolerance(positions[index], tree, feature_scores,
                                                                     feature_maximum)
            # if more than one grid point, search for the ones with the biggest tolerance
            else:
                core_tolerance, indices_maximal_tolerance, feature_indices_list = \
                    get_maximal_core_tolerance(indices, positions, tree, feature_scores, feature_maximum)
                # if more than one grid point with biggest tolerance, search for the one with the biggest score
                if len(indices_maximal_tolerance) > 1:
                    index, feature_indices = get_maximal_sum_of_scores(feature_scores, indices_maximal_tolerance,
                                                                       feature_indices_list)
                else:
                    index = indices_maximal_tolerance[0]
                    feature_indices = feature_indices_list[0]
            if len(feature_indices) + len(used) > len(set(feature_indices + used)):
                not_used = [x for x in not_used if x != index]
            else:
                generated_features.append([index, feature_type, 'M', positions[index], core_tolerance,
                                           get_partner_positions(feature_type, partners[index]),
                                           get_partner_tolerance(feature_type, core_tolerance), feature_scores[index]])
                not_used = [x for x in not_used if x not in feature_indices]
                used += feature_indices
                with feature_counter.get_lock():
                    feature_counter.value += 1
                update_progress(feature_counter.value / total_number_of_features, 'Progress of feature generation',
                                ((time.time() - start) / feature_counter.value) * (total_number_of_features -
                                                                                   feature_counter.value))
            if len(generated_features) >= features_per_feature_type:
                break
    if len(generated_features) < features_per_feature_type:
        with feature_counter.get_lock():
            feature_counter.value += features_per_feature_type - len(generated_features)
        update_progress(feature_counter.value / total_number_of_features, 'Progress of feature generation',
                        ((time.time() - start) / feature_counter.value) * (total_number_of_features -
                                                                           feature_counter.value))
    results += generated_features
    return


def generate_library(pharmacophore_path, output_format, library_dict, library_path, make_mandatory, pyrod_pharmacophore,
                     weight, directory, debugging):
    """ This function writes a combinatorial pharmacophore library. """
    logger = setup_logger('library', directory, debugging)
    update_user('Starting library generation.', logger)
    super_pharmacophore = pharmacophore_reader(pharmacophore_path, pyrod_pharmacophore, logger)
    pharmacophore_library = []
    essential_hb, essential_hi, essential_ai, essential_ii = [], [], [], []
    optional_hb, optional_hi, optional_ai, optional_ii = [], [], [], []
    exclusion_volumes = []
    # analyzing pharmacophore
    for index, feature in enumerate(super_pharmacophore):
        if feature[1] == 'ev':
            exclusion_volumes.append(feature)
        else:
            if feature[1] in ['ha', 'hd', 'ha2', 'hd2', 'hda']:
                if feature[2] == 'O':
                    optional_hb.append(index)
                else:
                    essential_hb.append(index)
            elif feature[1] == 'hi':
                if feature[2] == 'O':
                    optional_hi.append(index)
                else:
                    essential_hi.append(index)
            elif feature[1] in ['pi', 'ni']:
                if feature[2] == 'O':
                    optional_ii.append(index)
                else:
                    essential_ii.append(index)
            elif feature[1] == 'ai':
                if feature[2] == 'O':
                    optional_ai.append(index)
                else:
                    essential_ai.append(index)
    for hbs in [combinations(optional_hb, x) for x in
                range(library_dict['minimal hydrogen bonds'],
                      library_dict['maximal hydrogen bonds'] + 1)]:
        for hbs_ in hbs:
            for his in [combinations(optional_hi, x) for x in
                        range(library_dict['minimal hydrophobic interactions'],
                              library_dict['maximal hydrophobic interactions'] + 1)]:
                for his_ in his:
                    for ais in [combinations(optional_ai, x) for x in
                                range(library_dict['minimal aromatic interactions'],
                                      library_dict['maximal aromatic interactions'] + 1)]:
                        for ais_ in ais:
                            for iis in [combinations(optional_ii, x) for x in
                                        range(library_dict['minimal ionizable interactions'],
                                              library_dict['maximal ionizable interactions'] + 1)]:
                                for iis_ in iis:
                                    pharmacophore = (essential_hb + list(hbs_) +
                                                     essential_hi + list(his_) +
                                                     essential_ai + list(ais_) +
                                                     essential_ii + list(iis_))
                                    if evaluate_pharmacophore(pharmacophore, super_pharmacophore, library_dict,
                                                              pyrod_pharmacophore):
                                        pharmacophore_library.append(pharmacophore)
    # estimate maximal library size and ask user if number and space of pharmacophores is okay
    pharmacophore_writer(super_pharmacophore, [output_format], 'super_pharmacophore', library_path, weight, logger)
    pharmacophore_library_size = bytes_to_text(os.path.getsize('{}/{}.{}'.format(library_path, 'super_pharmacophore',
                                               output_format)) * len(pharmacophore_library))
    user_prompt = ''
    while user_prompt not in ['yes', 'no']:
        user_prompt = input('{} pharmacophores will be written taking about {} of space.\n'
                            'Do you want to continue? [yes/no]: '.format(len(pharmacophore_library),
                                                                         pharmacophore_library_size))
        if user_prompt == 'no':
            sys.exit()
    start = time.time()
    # write pharmacophores
    maximal_exclusion_volume_id = max([exclusion_volume[0] for exclusion_volume in exclusion_volumes])
    for counter, index_pharmacophore in enumerate(pharmacophore_library):
        extra_exclusion_volumes = []
        extra_ev_counter = 1
        pharmacophore = []
        for index_feature in index_pharmacophore:
            feature = super_pharmacophore[index_feature]
            if make_mandatory:
                feature[2] = 'M'
            pharmacophore.append(feature)
            if feature[1] in ['ha', 'hd', 'ha2', 'hd2', 'hda']:
                extra_exclusion_volumes.append([maximal_exclusion_volume_id + extra_ev_counter, 'ev', 'M',
                                               feature[5][0], 1.0, [], 0.0, 0.0])
                extra_ev_counter += 1
                if feature[1] in ['ha2', 'hd2', 'hda']:
                    extra_exclusion_volumes.append([maximal_exclusion_volume_id + extra_ev_counter, 'ev', 'M',
                                                   feature[5][1], 1.0, [], 0.0, 0.0])
                    extra_ev_counter += 1
        pharmacophore_writer(pharmacophore + exclusion_volumes + extra_exclusion_volumes, [output_format],
                             str(counter), library_path, weight, logger)
        update_progress((counter + 1) / len(pharmacophore_library),
                        'Writing {} pharmacophores'.format(len(pharmacophore_library)),
                        ((time.time() - start) / (counter + 1)) * (len(pharmacophore_library) - (counter + 1)))
    update_user('Wrote pharmacophores to {}.'.format(library_path), logger)
    return
