""" PyRod - dynamic molecular interaction fields (dMIFs), based on tracing water molecules in MD simulations.

This module contains helper functions used by the dmif module.
"""


# python standard library
import operator

# external libraries
import numpy as np
import numpy.lib.recfunctions as rfn

# pyrod modules
try:
    from pyrod.modules.lookup import protein_resnames, hb_types, hi_sel_dict, pi_sel_dict, ni_sel_dict, ai_sel_dict, \
        grid_score_dict, grid_list_dict
    from pyrod.modules.helper_math import angle, maximal_angle
except ImportError:
    from modules.lookup import protein_resnames, hb_types, hi_sel_dict, pi_sel_dict, ni_sel_dict, ai_sel_dict, \
        grid_score_dict, grid_list_dict
    from modules.helper_math import angle, maximal_angle


def grid_generator(center, edge_lengths, space):
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


def dmif_data_structure(grid):
    """ This function generates the central data structure for analyzing the trajectories. Returned will be a grid as
    numpy structured array whose first three columns are the coordinates and whose other columns are used for holding
    scores in later trajectory analysis. Additionally, a list of lists of lists will be returned with the same length
    as the grid that is used to save coordinates of interaction partners for e.g. hydrogen bonds. """
    grid_score = []
    grid_partners = []
    for position in grid:
        grid_score.append(position + [0] * (len(grid_score_dict.keys()) - 3))
        grid_partners.append([[] for _ in range(len(grid_list_dict.keys()))])
    grid_score = np.array([tuple(x) for x in grid_score], dtype=[(x[0], 'float64') for x in sorted([[x,
                          grid_score_dict[x]] for x in grid_score_dict.keys()], key=operator.itemgetter(1))])
    return [grid_score, grid_partners]


def grid_parameters(positions):
    """ This function returns parameters based on the input grid. """
    x_minimum, x_maximum, y_minimum, y_maximum, z_minimum, z_maximum = [min(positions[:, 0]), max(positions[:, 0]),
                                                                        min(positions[:, 1]), max(positions[:, 1]),
                                                                        min(positions[:, 2]), max(positions[:, 2])]
    space = positions[1][0] - positions[0][0]
    return [x_minimum, x_maximum, y_minimum, y_maximum, z_minimum, z_maximum, space]


def select_protein(topology):
    """ This function returns all protein atoms from a topology. """
    selection = ''
    for resname in protein_resnames:
        if len(selection) == 0:
            selection = topology[(topology['resname'] == resname)]
        else:
            selection = np.concatenate((selection, topology[(topology['resname'] == resname)]), axis=0)
    if len(selection) > 1:
        selection.sort(order='atomid')
    return selection


def select_hb_atoms(protein):
    """ This function returns all atoms with hydrogen bond capacity from a protein. """
    selection = ''
    for hb_type in hb_types:
        for resname in protein_resnames:
            if len(selection) == 0:
                selection = protein[(protein['resname'] == resname) & (protein['type'] == hb_type)]
            else:
                selection = np.concatenate((selection, protein[(protein['resname'] == resname) &
                                                               (protein['type'] == hb_type)]), axis=0)
    if len(selection) > 1:
        selection.sort(order='atomid')
    return selection


def select_hi_atoms(protein):
    """ This function returns all hydrophobic atoms from a protein. """
    selection = ''
    for resname in hi_sel_dict.keys():
        for name in hi_sel_dict[resname]:
            if len(selection) == 0:
                selection = protein[(protein['resname'] == resname) & (protein['name'] == name)]
            else:
                selection = np.concatenate((selection, protein[(protein['resname'] == resname) &
                                                               (protein['name'] == name)]), axis=0)
    cys_sel = protein[(protein['resname'] == 'CYS') & ((protein['name'] == 'CB') | (protein['name'] == 'SG'))]
    for cys_atom in cys_sel:
        if len(protein[((protein['resid'] == cys_atom['resid']) & (protein['name'] == 'HG'))]) == 0:
            if len(selection) == 0:
                selection = cys_atom
            else:
                selection = np.concatenate((selection, np.array([cys_atom])), axis=0)
    if len(selection) > 1:
        selection.sort(order='atomid')
    return selection


def select_pi_atoms(protein):
    """ This function returns all negatively charged atoms from a protein. """
    selection = ''
    for resname in pi_sel_dict.keys():
        for name in pi_sel_dict[resname]:
            if len(selection) == 0:
                selection = protein[(protein['resname'] == resname) & (protein['name'] == name)]
            else:
                selection = np.concatenate((selection, protein[(protein['resname'] == resname) &
                                                               (protein['name'] == name)]), axis=0)
    # need to check protonation state
    if len(selection) > 1:
        selection.sort(order='atomid')
    return selection


def select_ni_atoms(protein):
    """ This function returns all positively charged atoms from a protein. """
    selection = ''
    for resname in ni_sel_dict.keys():
        for name in ni_sel_dict[resname]:
            if len(selection) == 0:
                selection = protein[(protein['resname'] == resname) & (protein['name'] == name)]
                selection = np.concatenate((selection, protein[(protein['resname'] == resname) &
                                                               (protein['name'] == name)]), axis=0)
            else:
                selection = np.concatenate((selection, protein[(protein['resname'] == resname) &
                                                               (protein['name'] == name)]), axis=0)
                selection = np.concatenate((selection, protein[(protein['resname'] == resname) &
                                                               (protein['name'] == name)]), axis=0)
    # histidine protonations
    his_sel = protein[((protein['resname'] == 'HIS') | (protein['resname'] == 'HSD') | (protein['resname'] == 'HSE') |
                       (protein['resname'] == 'HSP')) & ((protein['name'] == 'ND1') | (protein['name'] == 'NE2'))]
    for his_atom in his_sel:
        if len(protein[(protein['resid'] == his_atom['resid']) & ((protein['name'] == 'HD1') |
                                                                  (protein['name'] == 'HE2'))]) == 2:
            if len(selection) == 0:
                selection = his_atom
            else:
                selection = np.concatenate((selection, np.array([his_atom])), axis=0)
    if len(selection) > 1:
        selection.sort(order='atomid')
    return selection


def select_ai_atoms(protein):
    """ This function returns atoms for defining aromatic centers from a topology. """
    selection = ''
    for resname in ai_sel_dict.keys():
        for aromatic_ring in ai_sel_dict[resname]:
            for name in aromatic_ring:
                if len(selection) == 0:
                    selection = protein[(protein['resname'] == resname) & (protein['name'] == name)]
                else:
                    selection = np.concatenate((selection, protein[(protein['resname'] == resname) &
                                                                   (protein['name'] == name)]), axis=0)
    if len(selection) > 1:
        selection.sort(order='atomid')
    return selection


def select_metal_atoms(topology, metal_names):
    """ This function returns metal atoms from a topology. """
    selection = ''
    for metal_name in metal_names:
        if len(selection) == 0:
            selection = topology[topology['resname'] == metal_name]
        else:
            selection = np.concatenate((selection, topology[topology['resname'] == metal_name]), axis=0)
    if len(selection) > 1:
        selection.sort(order='atomid')
    return selection


def position_angle_test(position_to_test, center_position, positions, cutoff):
    """ This function returns True if all possible angles between the position_to_test, center_position and all
    positions are equal or bigger than the specified cutoff. """
    test = True
    while test:
        for position in positions:
            if angle(position_to_test, center_position, position) < cutoff:
                test = False
                break
        break
    return test


def buriedness(center_position, positions):
    """ This function returns the buriedness score of a center atom based on surrounding atom positions. """
    angle_cutoff = 30
    score = 0
    positions_used = []
    origin_position = None
    while len(positions) > 0:
        if origin_position is None:
            angle_maximum, position_index, position_index_2 = maximal_angle(positions, center_position)
            if angle_maximum > angle_cutoff:
                origin_position = positions[position_index]
                positions_used.append(positions[position_index])
                positions_used.append(positions[position_index_2])
                positions = np.delete(positions, [position_index, position_index_2], 0)
                score += 1
            else:
                return score
        else:
            angle_maximum, position_index = maximal_angle(positions, center_position, origin_position)
            position_to_test = positions[position_index]
            if position_angle_test(position_to_test, center_position, positions_used, angle_cutoff):
                positions_used.append(position_to_test)
                score += 1
            positions = np.delete(positions, position_index, 0)
    return score


def dmif_processing(results, traj_number, length):
    dmif = results[0][0]
    partners = results[0][1]
    if len(results) > 1:
        for result in results[1:]:
            for feature_name in [x for x in dmif.dtype.names if x not in ['x', 'y', 'z']]:
                dmif[feature_name] += result[0][feature_name]
            for partner_name in partners.dtype.names:
                partners[partner_name] += result[1][partner_name]
    for feature_name in [x for x in dmif.dtype.names if x not in ['x', 'y', 'z']]:
        dmif[feature_name] = ((dmif[feature_name] * 100) / (traj_number * length))
    dmif['ni'] = np.clip(dmif['ni'], 0, None)
    dmif['pi'] = np.clip(dmif['pi'], 0, None)
    dmif['hi_hb'] = np.divide(dmif['hi_hb'], dmif['shape'], where=dmif['shape'] >= 1)
    dmif['hi_hb'][dmif['shape'] < 1] = 0
    hb = np.array(dmif['hd'] + dmif['hd2'] + dmif['ha'] + dmif['ha2'] + dmif['hda'], dtype=[('hb', 'float64')])
    dmif = rfn.merge_arrays([dmif, hb], flatten=True, usemask=False)
    hd_combo = np.array(dmif['hd'] + dmif['hd2'] + dmif['hda'], dtype=[('hd_combo', 'float64')])
    dmif = rfn.merge_arrays([dmif, hd_combo], flatten=True, usemask=False)
    ha_combo = np.array(dmif['ha'] + dmif['ha2'] + dmif['hda'], dtype=[('ha_combo', 'float64')])
    dmif = rfn.merge_arrays([dmif, ha_combo], flatten=True, usemask=False)
    return dmif, partners


def grid_partners_to_array(grid_partners):
    grid_partners = np.array([tuple(x) for x in grid_partners], dtype=[(x[0], 'O') for x in
                             sorted([[x, grid_list_dict[x]] for x in grid_list_dict.keys()],
                             key=operator.itemgetter(1))])
    return grid_partners
