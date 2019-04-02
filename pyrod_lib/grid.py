""" PyRod - dynamic molecular interaction fields (dMIFs), based on tracing water molecules in MD simulations.

This module contains functions needed to generate and process grid data structures.
"""


# python standard library
import copy
import operator

# external libraries
import pickle
import sys

import numpy as np

# pyrod modules
from numpy.lib import recfunctions as rfn

try:
    from pyrod.pyrod_lib.lookup import grid_score_dict, grid_list_dict
except ImportError:
    from pyrod_lib.lookup import grid_score_dict, grid_list_dict, feature_types


def grid_characteristics(positions):
    """ This function returns parameters based on the input grid. """
    x_minimum, x_maximum, y_minimum, y_maximum, z_minimum, z_maximum = [min(positions[:, 0]), max(positions[:, 0]),
                                                                        min(positions[:, 1]), max(positions[:, 1]),
                                                                        min(positions[:, 2]), max(positions[:, 2])]
    space = positions[1][0] - positions[0][0]
    return [x_minimum, x_maximum, y_minimum, y_maximum, z_minimum, z_maximum, space]


def generate_grid(center, edge_lengths, space=0.5):
    """ This function generates rectangular grids as list of lists. The variable space defines the distance between the
    grid points. The center of the grid and the length of the grid edges can be defined by center and edge_lengths. """
    x = min_x = center[0] - (edge_lengths[0] / 2)
    y = min_y = center[1] - (edge_lengths[1] / 2)
    z = center[2] - (edge_lengths[2] / 2)
    max_x = center[0] + (edge_lengths[0] / 2)
    max_y = center[1] + (edge_lengths[1] / 2)
    max_z = center[2] + (edge_lengths[2] / 2)
    grid = []
    while z <= max_z:
        while y <= max_y:
            while x <= max_x:
                grid.append([x, y, z])
                x += space
            y += space
            x = min_x
        z += space
        y = min_y
    return grid


def dmif_data_structure(grid, get_partners):
    """ This function generates the central data structure for dmif analysis of trajectories. Returned will be a grid as
    numpy structured array whose first three columns are the coordinates and whose other columns are used for holding
    scores in later trajectory analysis. Additionally, a list of lists of lists will be returned with the same length
    as the grid that is used to save coordinates of interaction partners for e.g. hydrogen bonds. """
    grid_score = []
    grid_partners = []
    for position in grid:
        grid_score.append(position + [0] * (len(grid_score_dict.keys()) - 3))
        if get_partners:
            grid_partners.append([[] if x[0] != 'hda' else [[], []] for x in sorted([[x, grid_list_dict[x]] for x in
                                                                                     grid_list_dict.keys()],
                                                                                    key=operator.itemgetter(1))])
    grid_score = np.array([tuple(x) for x in grid_score], dtype=[(x[0], float) for x in sorted([[x,
                          grid_score_dict[x]] for x in grid_score_dict.keys()], key=operator.itemgetter(1))])
    return [grid_score, grid_partners]


def grid_partners_to_array(grid_partners):
    grid_partners = np.array([tuple(x) for x in grid_partners], dtype=[(x[0], object) for x in
                             sorted([[x, grid_list_dict[x]] for x in grid_list_dict.keys()],
                             key=operator.itemgetter(1))])
    return grid_partners


def post_processing(results, total_number_of_frames):
    dmif = results[0][0]
    partners = results[0][1]
    if len(results) > 1:
        for result in results[1:]:
            for feature_name in [x for x in dmif.dtype.names if x not in ['x', 'y', 'z']]:
                dmif[feature_name] += result[0][feature_name]
            for partner_name in partners.dtype.names:
                if partner_name != 'hda':
                    partners[partner_name] += result[1][partner_name]
                else:
                    for counter in range(len(partners)):
                        partners[partner_name][counter][0] += result[1][partner_name][counter][0]
                        partners[partner_name][counter][1] += result[1][partner_name][counter][1]
    for feature_name in [x for x in dmif.dtype.names if x not in ['x', 'y', 'z']]:
        dmif[feature_name] = ((dmif[feature_name] * 100) / total_number_of_frames)
    dmif['ni'] = np.clip(dmif['ni'], 0, None)
    dmif['pi'] = np.clip(dmif['pi'], 0, None)
    dmif['hi_norm'] = np.divide(dmif['hi_norm'], dmif['shape'], where=dmif['shape'] >= 1)
    dmif['hi_norm'][dmif['shape'] < 1] = 0
    hb = np.array(dmif['hd'] + dmif['hd2'] + dmif['ha'] + dmif['ha2'] + dmif['hda'], dtype=[('hb', float)])
    dmif = rfn.merge_arrays([dmif, hb], flatten=True, usemask=False)
    hd_combo = np.array(dmif['hd'] + dmif['hd2'] + dmif['hda'], dtype=[('hd_combo', float)])
    dmif = rfn.merge_arrays([dmif, hd_combo], flatten=True, usemask=False)
    ha_combo = np.array(dmif['ha'] + dmif['ha2'] + dmif['hda'], dtype=[('ha_combo', float)])
    dmif = rfn.merge_arrays([dmif, ha_combo], flatten=True, usemask=False)
    return dmif, partners


def generate_dmif_excess(dmif1_path, dmif2_path):
    with open(dmif1_path, 'rb') as file:
        dmif1 = pickle.load(file)
    with open(dmif2_path, 'rb') as file:
        dmif2 = pickle.load(file)
    if np.array_equal(np.array([dmif1['x'], dmif1['y'], dmif1['z']]), np.array([dmif2['x'], dmif2['y'], dmif2['z']])):
        dmif1_excess = copy.deepcopy(dmif1)
        dmif2_excess = copy.deepcopy(dmif2)
        for feature_name in feature_types:
            dmif1_excess[feature_name] -= dmif2[feature_name]
            dmif1_excess[feature_name] = np.clip(dmif1_excess[feature_name], 0, None)
            dmif2_excess[feature_name] -= dmif1[feature_name]
            dmif2_excess[feature_name] = np.clip(dmif2_excess[feature_name], 0, None)
        return dmif1_excess, dmif2_excess
    else:
        print('Specified dmifs were not generated with the same grid parameters.')
        sys.exit()