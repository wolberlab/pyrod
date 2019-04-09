""" PyRod - dynamic molecular interaction fields (dMIFs), based on tracing water molecules in MD simulations.

This module contains helper functions needed to generate and process dmifs.
"""


# external libraries
import numpy as np

# pyrod modules
try:
    from pyrod.pyrod_lib.lookup import protein_resnames, ha_sel_dict, hd_sel_dict, hi_sel_dict, ni_sel_dict, \
        pi_sel_dict, ai_sel_dict, grid_score_dict, grid_list_dict, feature_types, standard_resnames_dict, \
        standard_atomnames_dict, valid_resnames
    from pyrod.pyrod_lib.math import angle, maximal_angle, norm, adjacent, vector, vector_angle, rotate_vector
    from pyrod.pyrod_lib.write import update_user
except ImportError:
    from pyrod_lib.lookup import protein_resnames, ha_sel_dict, hd_sel_dict, hi_sel_dict, ni_sel_dict, \
        pi_sel_dict, ai_sel_dict, grid_score_dict, grid_list_dict, feature_types, standard_resnames_dict, \
        standard_atomnames_dict, valid_resnames
    from pyrod_lib.math import angle, maximal_angle, norm, adjacent, vector, vector_angle, rotate_vector
    from pyrod_lib.write import update_user


def main_selection(topology):
    """ This function returns all atoms considered for interactions from a topology. """
    # renumber resids and standardize resnames and atomnames for protein
    counter = -1
    for atom in topology:
        # standardize resnames
        for alternative_names in standard_resnames_dict.keys():
            if atom['resname'] in alternative_names:
                atom['resname'] = standard_resnames_dict[alternative_names]
        # standardize atomnames
        if atom['resname'] in standard_atomnames_dict.keys():
            for alternative_names in standard_atomnames_dict[atom['resname']]:
                if atom['name'] in alternative_names:
                    atom['name'] = standard_atomnames_dict[atom['resname']][alternative_names]
        # renumber resids for protein
        if atom['resname'] in protein_resnames:
            if atom['name'] == 'N':
                counter += 1
        atom['resid'] = counter
    selection = np.empty((0, 0))
    for resname in valid_resnames:
        if selection.size == 0:
            selection = topology[(topology['resname'] == resname)]
        else:
            selection = np.concatenate((selection, topology[(topology['resname'] == resname)]), axis=0)
    if selection.size > 1:
        selection.sort(order='atomid')
    return selection


def heavy_atom_selection(main_atoms):
    """ This function returns all atomids for heavy atoms from the main selection. """
    atomids = []
    for atom in main_atoms:
        if atom['type'] != 'H':
            atomids.append(atom['atomid'])
    return np.array(atomids)


def hd_selection(main_atoms):
    """ This function returns all hydrogen bond donor atomids, their element types and the hydrogen atom ids for each
    donor from the main selection. """
    atomids = []
    types = []
    hydrogen_atomid_lists = []
    for atom in main_atoms:
        name = atom['name']
        resid = atom['resid']
        resname = atom['resname']
        if name in hd_sel_dict[resname].keys():
            hydrogen_atomids = []
            for hydrogen_name in hd_sel_dict[resname][name]:
                hydrogen_atomids += list(main_atoms[(main_atoms['resid'] == resid) &
                                                    (main_atoms['name'] == hydrogen_name)]['atomid'])
            if len(hydrogen_atomids) > 0:
                atomids.append(atom['atomid'])
                types.append(atom['type'])
                hydrogen_atomid_lists += [hydrogen_atomids]
    return np.array(atomids), np.array(types), np.array(hydrogen_atomid_lists)


def ha_selection(main_atoms):
    """ This function returns all hydrogen bond acceptor atomids from the main selection. """
    atomids = []
    types = []
    for atom in main_atoms:
        ha = False
        name = atom['name']
        resid = atom['resid']
        resname = atom['resname']
        if name in ha_sel_dict[resname]:
            if resname != 'HIS':
                ha = True
            else:
                if name in ['ND1', 'NE2']:
                    if len(main_atoms[(main_atoms['resid'] == resid) &
                                      (main_atoms['name'] == hd_sel_dict[resname][name])]) == 0:
                        ha = True
                else:
                    ha = True
        if ha:
            atomids.append(atom['atomid'])
            types.append(atom['type'])
    return np.array(atomids), np.array(types)


def hi_selection(main_atoms):
    """ This function returns all hydrophobic atomids from the main selection. """
    atomids = []
    for atom in main_atoms:
        hi = False
        name = atom['name']
        resid = atom['resid']
        resname = atom['resname']
        if resname in hi_sel_dict.keys():
            if name in hi_sel_dict[resname]:
                if resname != 'CYS':
                    hi = True
                else:
                    if len(main_atoms[(main_atoms['resid'] == resid) & (main_atoms['name'] == 'HG')]) == 1:
                        hi = True
        if hi:
            atomids.append(atom['atomid'])
    return np.array(atomids)


def ni_selection(main_atoms):
    """ This function returns all atomids from the main selection involved in a negative charge. """
    atomids = []
    for resid in set(main_atoms['resid']):
        residue_atoms = main_atoms[(main_atoms['resid']) == resid]
        resname = residue_atoms['resname'][0]
        if resname in ni_sel_dict.keys():
            for group_counter, group in enumerate(ni_sel_dict[resname]):
                charged = True
                hydrogen_counter = 0
                for hydrogen_name in group[1]:
                    if len(main_atoms[(main_atoms['resid'] == resid) & (main_atoms['name'] == hydrogen_name)]) > 0:
                        hydrogen_counter += 1
                if hydrogen_counter == len(group[1]):
                    charged = False
                if charged:
                    indices_tmp = []
                    for name in ni_sel_dict[resname][group_counter][0]:
                        index = main_atoms[(main_atoms['resid'] == resid) & (main_atoms['name'] == name)]['atomid']
                        if index.size == 1:
                            indices_tmp.append(index[0])
                    if len(indices_tmp) == len(ni_sel_dict[resname][group_counter][0]):
                        atomids += indices_tmp
                        if len(ni_sel_dict[resname][group_counter][0]) == 1:
                            atomids += indices_tmp
    return np.array(atomids)


def pi_selection(main_atoms):
    """ This function returns all atomids from the main selection involved in a positive charge. """
    atomids = []
    for resid in set(main_atoms['resid']):
        residue_atoms = main_atoms[(main_atoms['resid']) == resid]
        resname = residue_atoms['resname'][0]
        if resname in pi_sel_dict.keys():
            for group_counter, group in enumerate(pi_sel_dict[resname]):
                charged = False
                hydrogen_counter = 0
                for hydrogen_name in group[1]:
                    if len(main_atoms[(main_atoms['resid'] == resid) & (main_atoms['name'] == hydrogen_name)]) > 0:
                        hydrogen_counter += 1
                if hydrogen_counter == len(group[1]):
                    charged = True
                if charged:
                    indices_tmp = []
                    for name in pi_sel_dict[resname][group_counter][0]:
                        index = main_atoms[(main_atoms['resid'] == resid) & (main_atoms['name'] == name)]['atomid']
                        if index.size == 1:
                            indices_tmp.append(index[0])
                    if len(indices_tmp) == len(pi_sel_dict[resname][group_counter][0]):
                        atomids += indices_tmp
                        if len(pi_sel_dict[resname][group_counter][0]) == 1:
                            atomids += indices_tmp
    return np.array(atomids)


def ai_selection(main_atoms):
    """ This function returns atomids for defining aromatic centers from the main selection. """
    atomids = []
    for resid in set(main_atoms['resid']):
        residue_atoms = main_atoms[(main_atoms['resid']) == resid]
        resname = residue_atoms['resname'][0]
        if resname in ai_sel_dict.keys():
            for group_counter, group in enumerate(ai_sel_dict[resname]):
                charged = False
                indices_tmp = []
                if resname == 'HIS':
                    if len(main_atoms[(main_atoms['resid'] == resid) & ((main_atoms['name'] == 'HD1') |
                                                                        (main_atoms['name'] == 'HE2'))]) != 1:
                        charged = True
                if not charged:
                    for name in ai_sel_dict[resname][group_counter]:
                        index = main_atoms[(main_atoms['resid'] == resid) & (main_atoms['name'] == name)]['atomid']
                        if index.size == 1:
                            indices_tmp.append(index[0])
                    if len(indices_tmp) == len(ai_sel_dict[resname][group_counter]):
                        atomids += indices_tmp
    return np.array(atomids)


def metal_selection(topology, metal_names):
    """ This function returns atomids of metal atoms from a topology. """
    atomids = []
    for metal_name in metal_names:
        atomids += list(topology[topology['name'] == metal_name]['atomid'])
    return np.array(atomids)


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


def ai_geometry(AB, AC):
    """ This function returns characteristics of a triangle important for ai calculation. """
    alpha = vector_angle(AB, AC)
    if alpha > 90:
        AC = [-1 * x for x in AC]
        alpha = 180 - alpha
    return [AC, alpha]


def pi_stacking_partner_position(B, AC, c, alpha):
    """ This function returns the position of an interacting aromatic center for pi-stacking. """
    b = norm(AC)
    b_new = adjacent(alpha, c)
    position = [[float(x - ((y / b) * b_new)) for x, y in zip(B, AC)]]
    return [item for sublist in position for item in sublist]


def t_stacking_partner_position(A, B, AC, a, c, alpha, radial=False):
    """ This function returns the positions of interacting aromatic centers for t-stacking. """
    b = norm(AC)
    b_new = adjacent(alpha, c)
    if radial:
        C = [x + ((y / b) * b_new) for x, y in zip(A, AC)]
        BC = vector(B, C)
        vectors = [BC]
        vectors += [rotate_vector(BC, AC, x)for x in [30, 60, 90, 120, 150]]
        positions = [[float(x + ((y / a) * 3.5)) for x, y in zip(B, BC)] for BC in vectors]
        positions += [[float(x - ((y / a) * 3.5)) for x, y in zip(B, BC)] for BC in vectors]
        return [item for sublist in positions for item in sublist]
    else:
        position = [[float(x - ((y / b) * b_new)) for x, y in zip(B, AC)]]
        return [item for sublist in position for item in sublist]
