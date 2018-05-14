""" PyRod - dynamic molecular interaction fields (dMIFs), based on tracing water molecules in MD simulations.

This module analyzes molecular dynamics simulations and stores the results in arrays.
"""

# python standard libraries
import time
import logging
import warnings

# external libraries
import numpy as np
import MDAnalysis as mda
from scipy.spatial import cKDTree

# pyrod modules
try:
    from pyrod.modules.lookup import grid_list_dict, hb_dist_dict, hb_angl_dict, don_hydrogen_dict, acceptors, \
        sel_cutoff_dict
    from pyrod.modules.helper_dmif import select_protein, select_hb_atoms, select_hi_atoms, select_ni_atoms, \
        select_pi_atoms, select_metal_atoms, buriedness, grid_parameters, grid_partners_to_array
    from pyrod.modules.helper_math import distance, angle
    from pyrod.modules.helper_update import update_progress_dmif_parameters, update_progress_dmif
except ImportError:
    from modules.lookup import grid_list_dict, hb_dist_dict, hb_angl_dict, don_hydrogen_dict, acceptors, \
        sel_cutoff_dict
    from modules.helper_dmif import select_protein, select_hb_atoms, select_hi_atoms, select_ni_atoms, \
        select_pi_atoms, select_metal_atoms, buriedness, grid_parameters, grid_partners_to_array
    from modules.helper_math import distance, angle
    from modules.helper_update import update_progress_dmif_parameters, update_progress_dmif


def dmif(topology, trajectory, counter, length_trajectory, number_processes, number_trajectories, grid_score,
         grid_partners, first_frame, last_frame, water_name, metal_names):
    logger = logging.getLogger('_'.join(['trajectroy', str(counter + 1)]))
    check_progress, final, past_frames, future_frames = update_progress_dmif_parameters(
        counter, length_trajectory, number_processes, number_trajectories)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        u = mda.Universe(topology, trajectory)
    topology = np.array([(a, b, c, d, e) for a, b, c, d, e in
                         zip(range(len(u.atoms.resnames)), u.atoms.resnames, u.atoms.resids, u.atoms.names,
                             u.atoms.types)],
                        dtype=[('atomid', np.array([range(len(u.atoms.resnames))]).dtype),
                               ('resname', u.atoms.resnames.dtype), ('resid', u.atoms.resids.dtype),
                               ('name', u.atoms.names.dtype), ('type', u.atoms.types.dtype)])
    positions = np.array([[x, y, z] for x, y, z in zip(grid_score['x'], grid_score['y'], grid_score['z'])])
    x_minimum, x_maximum, y_minimum, y_maximum, z_minimum, z_maximum = grid_parameters(positions)[:-1]
    tree = cKDTree(positions)
    protein = select_protein(topology)
    hb_atoms = select_hb_atoms(protein)
    hi_atoms = select_hi_atoms(protein)
    pi_atoms = select_pi_atoms(protein)
    ni_atoms = select_ni_atoms(protein)
    metal_atoms = select_metal_atoms(topology, metal_names)
    # ai missing
    start = time.time()
    for frame, _ in enumerate(u.trajectory[first_frame:last_frame:]):
        # create index collectors
        shape_inds = []
        ha_inds = []
        ha2_inds = []
        hd_inds = []
        hd2_inds = []
        hda_inds = []
        tw_inds = []
        positions = u.atoms.positions
        h2os_os_box_inds = topology[((topology['resname'] == water_name) & (topology['name'] == 'O') &
                                    (positions[:, 0] >= x_minimum) & (positions[:, 0] <= x_maximum) &
                                    (positions[:, 1] >= y_minimum) & (positions[:, 1] <= y_maximum) &
                                    (positions[:, 2] >= z_minimum) & (positions[:, 2] <= z_maximum))]['atomid']
        if len(h2os_os_box_inds) > 0:
            tree_h2os = cKDTree(positions[h2os_os_box_inds])
            if len(hb_atoms) > 0:
                tree_hb = cKDTree(positions[hb_atoms['atomid']])
                hb_lists = tree_h2os.query_ball_tree(tree_hb, sel_cutoff_dict['hb'])
            else:
                hb_lists = [[]] * len(h2os_os_box_inds)
            if len(hi_atoms) > 0:
                tree_hi = cKDTree(positions[hi_atoms['atomid']])
                hi_lists = tree_h2os.query_ball_tree(tree_hi, sel_cutoff_dict['hi'])
            else:
                hi_lists = [[]] * len(h2os_os_box_inds)
            if len(pi_atoms) > 0:
                tree_pi = cKDTree([((x + y) / 2) for x, y in
                                  zip(positions[pi_atoms['atomid'][::2]], positions[pi_atoms['atomid'][1::2]])])
                pi_lists = tree_h2os.query_ball_tree(tree_pi, sel_cutoff_dict['ii'])
            else:
                pi_lists = [[]] * len(h2os_os_box_inds)
            if len(ni_atoms) > 0:
                tree_ni = cKDTree([((x + y) / 2) for x, y in
                                   zip(positions[ni_atoms['atomid'][::2]], positions[ni_atoms['atomid'][1::2]])])
                ni_lists = tree_h2os.query_ball_tree(tree_ni, sel_cutoff_dict['ii'])
            else:
                ni_lists = [[]] * len(h2os_os_box_inds)
            if len(metal_atoms) > 0:
                tree_metal = cKDTree(positions[metal_atoms['atomid']])
                metal_lists = tree_h2os.query_ball_tree(tree_metal, sel_cutoff_dict['metal'])
            else:
                metal_lists = [[]] * len(h2os_os_box_inds)
        else:
            h2os_os_box_inds, hb_lists, hi_lists, pi_lists, ni_lists, metal_lists = [], [], [], [], [], []
        for o_ind, hb_list, hi_list, pi_list, ni_list, metal_list in zip(h2os_os_box_inds, hb_lists, hi_lists,
                                                                         pi_lists, ni_lists, metal_lists):
            ha, ha_i, hd, hd_i, hi, pi, ni, ai, ai_i = 0, [], 0, [], 0, 0, 0, 0, []
            o_coor, h1_coor, h2_coor = positions[o_ind], positions[o_ind + 1], positions[o_ind + 2]
            # hydrogen bonds
            for hb_ind in hb_list:
                hb_atom = hb_atoms[hb_ind]
                hb_coor, hb_resname, hb_resid, hb_name, hb_type = [positions[hb_atom['atomid']], hb_atom['resname'],
                                                                   hb_atom['resid'], hb_atom['name'], hb_atom['type']]
                if distance(o_coor, hb_coor) <= hb_dist_dict[hb_type]:
                    if hb_name in acceptors:
                        for h_coor in [h1_coor, h2_coor]:
                            if angle(hb_coor, h_coor, o_coor) >= hb_angl_dict[hb_type]:
                                hd += 1
                                hd_i.append(hb_coor)
                    if hb_resname in don_hydrogen_dict.keys():
                        if hb_name in don_hydrogen_dict[hb_resname].keys():
                            for h_name in don_hydrogen_dict[hb_resname][hb_name]:
                                h_inds = protein[((protein['resname'] == hb_resname) & (protein['resid'] == hb_resid) &
                                                 (protein['name'] == h_name))]['atomid']
                                if len(h_inds) > 0:
                                    min_dis_ind = 0
                                    # if multiple hydrogen atoms met the selection criteria above, e.g. multimers
                                    if len(h_inds) > 1:
                                        min_dis = distance(hb_coor, positions[h_inds[0]])
                                        for i, h_ind in enumerate(h_inds[1:]):
                                            if min_dis > distance(hb_coor, positions[h_ind]):
                                                min_dis = distance(hb_coor, positions[h_ind])
                                                min_dis_ind = i + 1
                                    h_coor = positions[h_inds[min_dis_ind]]
                                    if angle(hb_coor, h_coor, o_coor) >= hb_angl_dict[hb_type]:
                                        ha += 1
                                        ha_i.append(hb_coor)
            for metal_ind in metal_list:
                metal_coor = positions[metal_atoms[metal_ind]['atomid']]
                ha += 1
                ha_i.append(metal_coor)
                ni += 2.6 / distance(o_coor, metal_coor)
            # indices of voxels close to water
            inds = tree.query_ball_point(o_coor, r=1.41)
            # trapped water molecules
            if hd + ha > 2:
                tw_inds += inds
            # water molecule is replaceable/displaceable
            else:
                # negative ionizable
                for ni_ind in ni_list:
                    ni += 2.6 / distance(o_coor, positions[ni_atoms[ni_ind]['atomid']])
                # positive ionizable
                for pi_ind in pi_list:
                    pi += 2.6 / distance(o_coor, positions[pi_atoms[pi_ind]['atomid']])
                # hydrophobic interactions
                if len(hi_list) > 0:
                    hi += 1
                    if len(hi_list) > 1:
                        hi += buriedness(o_coor, positions[hi_atoms[hi_list]['atomid']])
                # get grid points close to water molecule
                shape_inds += inds
                # check if water molecule is involved in any interactions
                if ha + hd + hi + pi + ni + ai > 0:
                    # adding score to grid
                    if hd == 0:
                        # single
                        if ha == 1:
                            ha_inds += inds
                            for ind in inds:
                                grid_partners[ind][grid_list_dict['ha_i']] += ha_i
                        # double
                        elif ha == 2:
                            ha2_inds += inds
                            for ind in inds:
                                grid_partners[ind][grid_list_dict['ha2_i']] += ha_i
                    # single hydrogen bond donors
                    elif hd == 1:
                        # single donor
                        if ha == 0:
                            hd_inds += inds
                            for ind in inds:
                                grid_partners[ind][grid_list_dict['hd_i']] += hd_i
                        # mixed donor acceptor
                        elif ha == 1:
                            hda_inds += inds
                            for ind in inds:
                                grid_partners[ind][grid_list_dict['hda_id']] += hd_i
                                grid_partners[ind][grid_list_dict['hda_ia']] += ha_i
                    else:
                        # double hydrogen bond donor
                        hd2_inds += inds
                        for ind in inds:
                            grid_partners[ind][grid_list_dict['hd2_i']] += hd_i
                    # ionizable interactions
                    if pi > 0:
                        grid_score['pi'][inds] += pi
                        grid_score['ni'][inds] -= pi
                    # positive ionizable, pi cation interaction missing
                    if ni > 0:
                        grid_score['ni'][inds] += ni
                        grid_score['pi'][inds] -= ni
                    # hydrophobic interaction
                    if hi > 0:
                        # 2.6 / 4 = 0.65 --> definitely not involved in a charged hydrogen bond
                        if ni < 0.65 > pi:
                            grid_score['hi'][inds] += hi
                        if ha + hd > 0:
                            grid_score['hi_hb'][inds] += hi
                        # else:
                        #     grid_score['hi'][inds] += hi
                    # aromatic interaction missing
        # adding scores to grid
        grid_score['shape'][shape_inds] += 1
        grid_score['ha'][ha_inds] += 1
        grid_score['ha2'][ha2_inds] += 1
        grid_score['hd'][hd_inds] += 1
        grid_score['hd2'][hd2_inds] += 1
        grid_score['hda'][hda_inds] += 1
        grid_score['tw'][tw_inds] += 1
        if check_progress:
            update_progress_dmif(counter, frame, length_trajectory, number_trajectories, number_processes, past_frames,
                                 future_frames, start, final)
        logger.info('Trajectory {} finished with frame {}.'.format(counter + 1, frame + 1))
    grid_partners = grid_partners_to_array(grid_partners)
    return [grid_score, grid_partners]