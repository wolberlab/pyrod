""" PyRod - dynamic molecular interaction fields (dMIFs), based on tracing water molecules in MD simulations.

This module contains helper functions used by the pharmacophore module.
"""


# python standard library
from itertools import combinations, product
import operator

# external libraries
from scipy.spatial import cKDTree

# pyrod modules
try:
    from pyrod.modules.helper_math import distance, standard_deviation
    from pyrod.modules.lookup import feature_names
    from pyrod.modules.helper_update import update_user
except ImportError:
    from modules.helper_math import distance, standard_deviation
    from modules.lookup import feature_names
    from modules.helper_update import update_user


def center(positions, cutoff):
    """ This function returns the position with the most neighbors within the specified cutoff. If multiple
    positions have the most neighbors, the position with the lowest standard deviation of the distances to its
    neighbors is returned. """
    tree = cKDTree(positions)
    indices_lists = tree.query_ball_tree(tree, cutoff)
    indices_maximal_neighbors = []
    maximal_neighbours = max([len(x) for x in indices_lists])
    for index, x in enumerate(indices_lists):
        if len(x) == maximal_neighbours:
            indices_maximal_neighbors.append(index)
    if len(indices_maximal_neighbors) > 1:
        minimal_stddev = None
        index_minimal_stddev = None
        counter = 0
        for index in indices_maximal_neighbors:
            stddev = standard_deviation(
                [distance(x, positions[index]) for x in [positions[x] for x in indices_lists[index]]])
            if minimal_stddev is None:
                minimal_stddev = stddev
                index_minimal_stddev = index
            elif stddev < minimal_stddev:
                minimal_stddev = stddev
                index_minimal_stddev = index
        return [positions[index_minimal_stddev], indices_lists[index_minimal_stddev]]
    else:
        return [positions[indices_maximal_neighbors[0]], indices_lists[indices_maximal_neighbors[0]]]


def feature_tolerance(position, tree, scores, maximal_score):
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


def maximal_feature_tolerance(indices, positions, tree, feature_scores, feature_maximum):
    """ This function returns the maximal tolerance for a given list of position indices and returns those indices
    that have the maximal tolerance. Additionally indices of involved positions of the tolerance sphere are returned.
    """
    final_tolerance = 0
    matching_indices = []
    involved_indices_list = []
    for index in indices:
        tolerance, involved_indices = feature_tolerance(positions[index], tree, feature_scores, feature_maximum)
        if tolerance > 0:
            if tolerance == final_tolerance:
                involved_indices_list.append(involved_indices)
                matching_indices.append(index)
            elif tolerance > final_tolerance:
                final_tolerance = tolerance
                involved_indices_list = [involved_indices]
                matching_indices = [index]
    return [final_tolerance, matching_indices, involved_indices_list]


def maximal_sum_of_scores(feature_scores, indices, feature_indices):
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


def generate_feature(feature_name, index, positions, partners, feature_scores, tolerance):
    feature_partners = []
    if feature_name in ['ha', 'hd', 'ha2', 'hd2']:
        partner, used_list = center(partners[feature_name + '_i'][index], 1.5)
        feature_partners.append(partner)
        if feature_name in ['ha2', 'hd2']:
            partner2 = center([x for y, x in enumerate(partners[feature_name + '_i'][index]) if y not in used_list],
                              1.5)[0]
            feature_partners.append(partner2)
    elif feature_name == 'hda':
        feature_partners.append(center(partners[feature_name + '_id'][index], 1.5)[0])
        feature_partners.append(center(partners[feature_name + '_ia'][index], 1.5)[0])
    return [feature_name, positions[index], tolerance, feature_partners, 0, round(feature_scores[index], 1)]


def features_processing(results, features_per_feature_type):
    features = results[0]
    if len(results) > 1:
        for result in results[1:]:
            features += result
    features_processed = []
    for feature_name in feature_names:
        features_processed += sorted([x for x in features if x[0] == feature_name], key=operator.itemgetter(5),
                                     reverse=True)[:features_per_feature_type]
    features_processed = [[counter] + x for counter, x in enumerate(features_processed)]
    return features_processed


def select_features(features, hbs_number, his_number, iis_number):
    hbs = sorted([x for x in features if x[1] in ['hd', 'ha', 'hd2', 'ha2', 'hda']],
                 key=operator.itemgetter(6), reverse=True)[:hbs_number]
    his = [x for x in features if x[1] in ['hi']][:his_number]
    iis = sorted([x for x in features if x[1] in ['pi', 'ni']], key=operator.itemgetter(6), reverse=True)[:iis_number]
    return hbs + his + iis


def evaluate_pharmacophore(pharmacophore, super_pharmacophore, minimal_features, maximal_features, pyrod_pharmacophore):
    hb_positions = []
    hi_positions = []
    ai_positions = []
    ii_positions = []
    positions = []
    for index in pharmacophore:
        position = None
        feature = super_pharmacophore[index]
        feature_name = feature.attrib['name']
        if feature_name in ['H', 'AR', 'PI', 'NI']:
            position = feature.find('position')
            position = [float(position.attrib['x3']), float(position.attrib['y3']), float(position.attrib['z3'])]
            if feature_name == 'H':
                hi_positions.append(position)
            elif feature_name == 'AR':
                ai_positions.append(position)
            else:
                ii_positions.append(position)
        elif feature_name in ['HBA', 'HBD']:
            if feature_name == 'HBA':
                position = feature.find('target')
            else:
                position = feature.find('origin')
            position = [float(position.attrib['x3']), float(position.attrib['y3']), float(position.attrib['z3'])]
            if pyrod_pharmacophore:
                if len(feature.attrib['featureId'].split('_')) > 1:
                    if feature.attrib['featureId'].split('_')[1] == '1':
                        hb_positions.append(position)
                else:
                    hb_positions.append(position)
            else:
                hb_positions.append(position)
        if position is not None:
            if position not in positions:
                positions.append(position)
    # number of independent features should not be lower than minimum
    if minimal_features > len(positions):
        return False
    # number of independent features should not be higher than maximum
    if maximal_features < len(positions):
        return False
    # hydrophobic features should not be within 3 A of ionizable features
    for pair in product(hi_positions, ii_positions):
        if distance(*pair) < 3:
            return False
    # different hydrogen bond types should not appear within 1.5 A if pyrod pharmacophore
    if pyrod_pharmacophore:
        for pair in combinations(hb_positions, 2):
            if distance(*pair) < 1.5:
                return False
    # different ionizable features should not appear within 3 A
    for pair in combinations(ii_positions, 2):
        if distance(*pair) < 3:
            return False
    return True
