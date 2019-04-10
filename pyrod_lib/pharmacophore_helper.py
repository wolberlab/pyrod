""" PyRod - dynamic molecular interaction fields (dMIFs), based on tracing water molecules in MD simulations.

Released under the GNU Public Licence v2.

This module contains helper functions mainly used by the pharmacophore module.
"""


# python standard library
from itertools import combinations, product
import operator

# external libraries
import numpy as np
from scipy.spatial import cKDTree

# pyrod modules
try:
    from pyrod.pyrod_lib.grid import grid_characteristics, generate_grid
    from pyrod.pyrod_lib.lookup import feature_types
    from pyrod.pyrod_lib.math import distance, standard_deviation
    from pyrod.pyrod_lib.write import update_user
except ImportError:
    from pyrod_lib.grid import grid_characteristics, generate_grid
    from pyrod_lib.lookup import feature_types
    from pyrod_lib.math import distance, standard_deviation
    from pyrod_lib.write import update_user


def get_center(positions, cutoff):
    """ This function returns the approximate position with the most neighbors within the specified cutoff. If multiple
    positions have the most neighbors, the position with the lowest standard deviation of the distances to its
    neighbors is returned. """
    positions = np.array([[x, y, z] for x, y, z in zip(positions[0::3], positions[1::3], positions[2::3])])
    x_minimum, x_maximum, y_minimum, y_maximum, z_minimum, z_maximum = grid_characteristics(positions)[:-1]
    x_center, y_center, z_center = [round((x_minimum + x_maximum) / 2, 1), round((y_minimum + y_maximum) / 2, 1),
                                    round((z_minimum + z_maximum) / 2, 1)]
    x_length, y_length, z_length = [round(x_maximum - x_minimum, 1), round(y_maximum - y_minimum, 1),
                                    round(z_maximum - z_minimum, 1)]
    grid = generate_grid([x_center, y_center, z_center], [x_length, y_length, z_length], 0.1)
    tree = cKDTree(positions)
    length_positions = len(positions)
    less_positions = [positions[x] for x in set(tree.query(grid, distance_upper_bound=0.1)[1]) if x != length_positions]
    small_tree = cKDTree(less_positions)
    indices_lists = small_tree.query_ball_tree(tree, cutoff)
    indices_maximal_neighbors = []
    maximal_neighbours = max([len(x) for x in indices_lists])
    for index, x in enumerate(indices_lists):
        if len(x) == maximal_neighbours:
            indices_maximal_neighbors.append(index)
    if len(indices_maximal_neighbors) > 1:
        minimal_stddev = None
        index_minimal_stddev = None
        for index in indices_maximal_neighbors:
            stddev = standard_deviation(
                [distance(x, less_positions[index]) for x in [positions[x] for x in indices_lists[index]]])
            if minimal_stddev is None:
                minimal_stddev = stddev
                index_minimal_stddev = index
            elif stddev < minimal_stddev:
                minimal_stddev = stddev
                index_minimal_stddev = index
        return [less_positions[index_minimal_stddev], indices_lists[index_minimal_stddev]]
    else:
        return [less_positions[indices_maximal_neighbors[0]], indices_lists[indices_maximal_neighbors[0]]]


def get_core_tolerance(position, tree, scores, maximal_score):
    """ This function returns the tolerance for a pharmacophore feature and the indices of involved positions. The
    tolerance is determined by checking the score of nearby positions. If the score of one of the positions within a
    certain radius is below half of the maximal score provided, the last checked radius will be returned as tolerance.
    """
    minimal_cutoff = 1.5
    maximal_cutoff = 3
    step_size = 0.5
    involved_indices = []
    for tolerance in [x / 10 for x in range(int(minimal_cutoff * 10), int((maximal_cutoff * 10) + 1),
                                            int(step_size * 10))]:
        involved_indices = tree.query_ball_point(position, r=tolerance)
        if scores[involved_indices].min() < (maximal_score / 2.0):
            return tolerance, tree.query_ball_point(position, r=tolerance)
    return [maximal_cutoff, involved_indices]


def get_maximal_core_tolerance(indices, positions, tree, feature_scores, feature_maximum):
    """ This function returns the maximal core tolerance for a given list of position indices and returns those indices
    that have the maximal tolerance. Additionally, indices of involved positions of the tolerance sphere are returned.
    """
    final_tolerance = 0
    matching_indices = []
    involved_indices_list = []
    for index in indices:
        tolerance, involved_indices = get_core_tolerance(positions[index], tree, feature_scores, feature_maximum)
        if tolerance > 0:
            if tolerance == final_tolerance:
                involved_indices_list.append(involved_indices)
                matching_indices.append(index)
            elif tolerance > final_tolerance:
                final_tolerance = tolerance
                involved_indices_list = [involved_indices]
                matching_indices = [index]
    return [final_tolerance, matching_indices, involved_indices_list]


def get_maximal_sum_of_scores(feature_scores, indices, feature_indices):
    """ This function identifies the index whose feature indices have the biggest sum of scores. The identified index
    with the feature indices are returned. """
    maximal_score = 0
    index_maximal_score = None
    involved_indices_maximal_score = []
    for index, involved_indices in zip(indices, feature_indices):
        if sum(feature_scores[involved_indices]) > maximal_score:
            maximal_score = sum(feature_scores[involved_indices])
            index_maximal_score = index
            involved_indices_maximal_score = involved_indices
    return [index_maximal_score, involved_indices_maximal_score]


def get_partner_tolerance(feature_type, core_tolerance):
    """ This function returns the partner tolerance for a pharmacophore feature with directionality. """
    partner_tolerance = 0.0
    if feature_type in ['ha', 'hd', 'ha2', 'hd2', 'hda']:
        partner_tolerance = core_tolerance * 1.2999999333333332
    elif feature_type == 'ai':
        partner_tolerance = 0.43633232
    return partner_tolerance


def get_partner_positions(feature_type, partner_positions_list):
    """ This function returns a list of partner positions for a pharmacophore feature with directionality. """
    partner_positions = []
    if feature_type in ['ha', 'hd', 'ha2', 'hd2', 'ai']:
        partner_position, used_list = get_center(partner_positions_list, 1.5)
        partner_positions.append(partner_position)
        if feature_type in ['ha2', 'hd2']:
            partner_position2 = get_center([x for y, x in enumerate(partner_positions_list) if y not in used_list], 1.5)[0]
            partner_positions.append(partner_position2)
    elif feature_type == 'hda':
        partner_positions.append(get_center(partner_positions_list[0], 1.5)[0])
        partner_positions.append(get_center(partner_positions_list[1], 1.5)[0])
    return partner_positions


def renumber_features(pharmacophore):
    """ This function renumbers features. """
    return [[counter + 1] + x[1:] for counter, x in enumerate(pharmacophore)]


def select_features(pharmacophore, hbs_number, his_number, iis_number, ais_number):
    """ This functions returns a list of best features per feature class. """
    hbs = sorted([feature for feature in pharmacophore if feature[1] in ['hd', 'ha', 'hd2', 'ha2', 'hda']],
                 key=operator.itemgetter(7), reverse=True)[:hbs_number]
    his = [feature for feature in pharmacophore if feature[1] in ['hi']][:his_number]
    iis = sorted([feature for feature in pharmacophore if feature[1] in ['pi', 'ni']], key=operator.itemgetter(6),
                 reverse=True)[:iis_number]
    ais = [feature for feature in pharmacophore if feature[1] in ['ai']][:ais_number]
    evs = [feature for feature in pharmacophore if feature[1] == 'ev']
    return hbs + his + iis + ais + evs


def evaluate_pharmacophore(pharmacophore, super_pharmacophore, library_dict, pyrod_pharmacophore):
    """ This function evaluates if a pharmacophore matches the pharmacophore library criteria. """
    positions = []
    hb_positions = []
    hb_count = 0
    hi_positions = []
    ai_positions = []
    ii_positions = []
    for index in pharmacophore:
        feature = super_pharmacophore[index]
        feature_type = feature[1]
        core_position = feature[3]
        if core_position not in positions:
            positions.append(core_position)
        if feature_type in ['ha', 'hd', 'ha2', 'hd2', 'hda']:
            hb_count += len(feature[5])
            hb_positions.append(core_position)
        elif feature_type == 'hi':
            hi_positions.append(core_position)
        elif feature_type == 'ai':
            ai_positions.append(core_position)
        elif feature_type in ['pi', 'ni']:
            ii_positions.append(core_position)
    # number of independent features should not be lower than minimum
    if library_dict['minimal features'] > len(positions):
        return False
    # number of independent features should not be higher than maximum
    if library_dict['maximal features'] < len(positions):
        return False
    # number of hydrogen bonds should not be lower than minimum
    if library_dict['minimal hydrogen bonds'] > hb_count:
        return False
    # number of hydrogen bonds should not be higher than maximum
    if library_dict['maximal hydrogen bonds'] < hb_count:
        return False
    # number of hydrophobic interactions should not be lower than minimum
    if library_dict['minimal hydrophobic interactions'] > len(hi_positions):
        return False
    # number of hydrophobic interactions should not be higher than maximum
    if library_dict['maximal hydrophobic interactions'] < len(hi_positions):
        return False
    # number of aromatic interactions should not be lower than minimum
    if library_dict['minimal aromatic interactions'] > len(ai_positions):
        return False
    # number of aromatic interactions should not be higher than maximum
    if library_dict['maximal aromatic interactions'] < len(ai_positions):
        return False
    # number of ionizable interactions should not be lower than minimum
    if library_dict['minimal ionizable interactions'] > len(ii_positions):
        return False
    # number of ionizable interactions should not be higher than maximum
    if library_dict['maximal ionizable interactions'] < len(ii_positions):
        return False
    if pyrod_pharmacophore:
        # hydrophobic features should not be within 3 A of ionizable features
        for pair in product(hi_positions, ii_positions):
            if distance(*pair) < 3:
                return False
        # different hydrogen bond features should not appear within 1.5 A
        for pair in combinations(hb_positions, 2):
            if distance(*pair) < 1.5:
                return False
        # different ionizable features should not appear within 3 A
        for pair in combinations(ii_positions, 2):
            if distance(*pair) < 3:
                return False
    return True
