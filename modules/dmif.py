""" PyRod - dynamic molecular interaction fields (dMIFs), based on tracing water molecules in MD simulations.

This module analyzes molecular dynamics simulations and stores the results in arrays.
"""

# python standard libraries
import time
import warnings

# external libraries
import numpy as np
import MDAnalysis as mda
from scipy.spatial import cKDTree

# pyrod modules
try:
    from pyrod.modules.lookup import grid_list_dict, hb_dist_dict, hb_angl_dict, don_hydrogen_dict, acceptors, \
        sel_cutoff_dict, pi_stacking_distance_score_dict, t_stacking_distance_score_dict, ai_pi_distance_score_dict, \
        AI_PI_ANGLE_CUTOFF
    from pyrod.modules.helper_dmif import select_protein, select_hb_atoms, select_hi_atoms, select_ni_atoms, \
        select_pi_atoms, select_ai_atoms, select_metal_atoms, buriedness, pi_stacking_partner_position, \
        grid_parameters, grid_partners_to_array, ai_geometry, t_stacking_partner_position
    from pyrod.modules.helper_math import distance, angle, normal, opposite, adjacent, norm, vector_angle, vector, \
        cross_product
    from pyrod.modules.helper_update import update_progress_dmif_parameters, update_progress_dmif
    from pyrod.modules.helper_write import setup_logger
except ImportError:
    from modules.lookup import grid_list_dict, hb_dist_dict, hb_angl_dict, don_hydrogen_dict, acceptors, \
        sel_cutoff_dict, pi_stacking_distance_score_dict, t_stacking_distance_score_dict, ai_pi_distance_score_dict, \
        AI_PI_ANGLE_CUTOFF
    from modules.helper_dmif import select_protein, select_hb_atoms, select_hi_atoms, select_ni_atoms, \
        select_pi_atoms, select_ai_atoms, select_metal_atoms, buriedness, pi_stacking_partner_position, \
        grid_parameters, grid_partners_to_array, ai_geometry, t_stacking_partner_position
    from modules.helper_math import distance, angle, normal, opposite, adjacent, norm, vector_angle, vector, \
        cross_product
    from modules.helper_update import update_progress_dmif_parameters, update_progress_dmif
    from modules.helper_write import setup_logger


def dmif(topology, trajectory, counter, length_trajectory, number_processes, number_trajectories, grid_score,
         grid_partners, first_frame, last_frame, water_name, metal_names, directory, debugging):
    logger = setup_logger('_'.join(['trajectory', str(counter + 1)]), directory, debugging)
    logger.info('Started analysis of trajectory {}.'.format(counter + 1))
    check_progress, final, past_frames, future_frames = update_progress_dmif_parameters(
        counter, length_trajectory, number_processes, number_trajectories)
    if debugging:
        u = mda.Universe(topology, trajectory)
    else:
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
    protein_atoms = select_protein(topology)
    hb_atoms = select_hb_atoms(protein_atoms)
    hi_atoms = select_hi_atoms(protein_atoms)
    pi_atoms = select_pi_atoms(protein_atoms)
    ni_atoms = select_ni_atoms(protein_atoms)
    ai_atoms = select_ai_atoms(protein_atoms)
    metal_atoms = select_metal_atoms(topology, metal_names)
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
                hb_positions = positions[hb_atoms['atomid']]
                hb_resnames = hb_atoms['resname']
                hb_resids = hb_atoms['resid']
                hb_names = hb_atoms['name']
                hb_types = hb_atoms['type']
                hb_lists = tree_h2os.query_ball_tree(cKDTree(hb_positions), sel_cutoff_dict['hb'])
            else:
                hb_lists = [[]] * len(h2os_os_box_inds)
            if len(hi_atoms) > 0:
                hi_positions = positions[hi_atoms['atomid']]
                hi_lists = tree_h2os.query_ball_tree(cKDTree(hi_positions), sel_cutoff_dict['hi'])
            else:
                hi_lists = [[]] * len(h2os_os_box_inds)
            if len(pi_atoms) > 0:
                pi_positions = [((x + y) / 2) for x, y in zip(positions[pi_atoms['atomid'][::2]],
                                                              positions[pi_atoms['atomid'][1::2]])]
                pi_lists = tree_h2os.query_ball_tree(cKDTree(pi_positions), sel_cutoff_dict['ii'])
            else:
                pi_lists = [[]] * len(h2os_os_box_inds)
            if len(ni_atoms) > 0:
                ni_positions = [((x + y) / 2) for x, y in zip(positions[ni_atoms['atomid'][::2]],
                                                              positions[ni_atoms['atomid'][1::2]])]
                ni_lists = tree_h2os.query_ball_tree(cKDTree(ni_positions), sel_cutoff_dict['ii'])
            else:
                ni_lists = [[]] * len(h2os_os_box_inds)
            if len(ai_atoms) > 0:
                ai_positions = [((x + y + z) / 3) for x, y, z in zip(positions[ai_atoms['atomid'][::3]],
                                                                     positions[ai_atoms['atomid'][1::3]],
                                                                     positions[ai_atoms['atomid'][2::3]])]
                ai_normals = [normal(a, b, c) for a, b, c in zip(positions[ai_atoms['atomid'][::3]], ai_positions,
                                                                 positions[ai_atoms['atomid'][2::3]])]
                ai_lists = tree_h2os.query_ball_tree(cKDTree(ai_positions), sel_cutoff_dict['ai'])
            else:
                ai_lists = [[]] * len(h2os_os_box_inds)
            if len(metal_atoms) > 0:
                metal_positions = positions[metal_atoms['atomid']]
                metal_lists = tree_h2os.query_ball_tree(cKDTree(metal_positions), sel_cutoff_dict['metal'])
            else:
                metal_lists = [[]] * len(h2os_os_box_inds)
        else:
            h2os_os_box_inds, hb_lists, hi_lists, pi_lists, ni_lists, ai_lists, metal_lists = [], [], [], [], [], [], []
        for o_ind, hb_list, hi_list, pi_list, ni_list, ai_list, metal_list in \
                zip(h2os_os_box_inds, hb_lists, hi_lists, pi_lists, ni_lists, ai_lists, metal_lists):
            ha, ha_i, hd, hd_i, hi, pi, ni, ai, ai_i, ai_n = 0, [], 0, [], 0, 0, 0, 0, [], []
            o_coor, h1_coor, h2_coor = positions[o_ind], positions[o_ind + 1], positions[o_ind + 2]
            # hydrogen bonds
            for hb_ind in hb_list:
                hb_coor, hb_resname, hb_resid, hb_name, hb_type = [hb_positions[hb_ind], hb_resnames[hb_ind],
                                                                   hb_resids[hb_ind], hb_names[hb_ind],
                                                                   hb_types[hb_ind]]
                if distance(o_coor, hb_coor) <= hb_dist_dict[hb_type]:
                    if hb_name in acceptors:
                        for h_coor in [h1_coor, h2_coor]:
                            if angle(hb_coor, h_coor, o_coor) >= hb_angl_dict[hb_type]:
                                hd += 1
                                hd_i.append(hb_coor)
                    if hb_resname in don_hydrogen_dict.keys():
                        if hb_name in don_hydrogen_dict[hb_resname].keys():
                            for h_name in don_hydrogen_dict[hb_resname][hb_name]:
                                h_inds = protein_atoms[((protein_atoms['resname'] == hb_resname) &
                                                        (protein_atoms['resid'] == hb_resid) &
                                                        (protein_atoms['name'] == h_name))]['atomid']
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
            # metals
            for metal_ind in metal_list:
                metal_position = metal_positions[metal_ind]
                ha += 1
                ha_i.append(metal_position)
                ni += 2.6 / distance(o_coor, metal_position)
            # indices of points close to water
            inds = tree.query_ball_point(o_coor, r=1.41)
            # trapped water molecules
            if hd + ha > 2:
                tw_inds += inds
            # water molecule is replaceable/displaceable
            else:
                # shape
                shape_inds += inds
                # hydrogen bond features
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
                # sum up negative ionizable
                for ni_ind in ni_list:
                    ni += 2.6 / distance(o_coor, ni_positions[ni_ind])
                # sum up positive ionizable
                for pi_ind in pi_list:
                    pi += 2.6 / distance(o_coor, pi_positions[pi_ind])
                # add ionizable interaction score
                if pi > 0:
                    grid_score['pi'][inds] += pi
                    grid_score['ni'][inds] -= pi
                if ni > 0:
                    grid_score['ni'][inds] += ni
                    grid_score['pi'][inds] -= ni
                # hydrophobic interactions
                if len(hi_list) > 0:
                    hi += 1
                    if len(hi_list) > 1:
                        hi += buriedness(o_coor, hi_positions[hi_list])
                if hi > 0:
                    # no charged amino acid within hydrogen bond distance
                    if ni < 0.65 > pi:
                        grid_score['hi'][inds] += hi
                        if ha + hd > 0:
                            grid_score['hi_hb'][inds] += hi
                # aromatic interactions
                for ai_ind in ai_list:
                    ai_i = ai_positions[ai_ind]
                    ai_n = ai_normals[ai_ind]
                    # pi-cation interactions
                    if round(distance(o_coor, ai_i), 1) >= 3.1:
                        ai_vector_angle = vector_angle(ai_n, vector(ai_i, o_coor))
                        if ai_vector_angle <= AI_PI_ANGLE_CUTOFF or ai_vector_angle >= (180 - AI_PI_ANGLE_CUTOFF):
                            grid_score['pi'][inds] += ai_pi_distance_score_dict[round(distance(o_coor, ai_i), 1)]
                    # pi- and t-stacking grid point wise
                    for ind in inds:
                        grid_point = [grid_score['x'][ind], grid_score['y'][ind], grid_score['z'][ind]]
                        ai_distance = distance(grid_point, ai_i)
                        # check distance between grid point and aromatic center
                        if 3.3 <= ai_distance <= 6.0:
                            ai_vector = vector(ai_i, grid_point)
                            ai_n, alpha = ai_geometry(ai_vector, ai_n)
                            # pi- and t-stacking with pi-system of protein aromatic center
                            if alpha < 45:
                                offset = opposite(alpha, ai_distance)
                                # pi-stacking
                                if ai_distance <= 4.7:
                                    # check offset between grid point and aromatic center
                                    if 0.001 <= offset <= 2.0:
                                        grid_score['ai'][ind] += pi_stacking_distance_score_dict[round(ai_distance, 1)]
                                        grid_partners[ind][grid_list_dict['ai_i']] += pi_stacking_partner_position(
                                                                                      grid_point, ai_n,
                                                                                      ai_distance, alpha)
                                # t-stacking
                                else:
                                    # check offset between grid point and aromatic center
                                    if 0.001 <= offset <= 0.5:
                                        grid_score['ai'][ind] += t_stacking_distance_score_dict[round(ai_distance, 1)]
                                        grid_partners[ind][grid_list_dict['ai_i']] += t_stacking_partner_position(
                                                                                      ai_i, grid_point, ai_n, offset,
                                                                                      ai_distance, alpha, True)
                            # t-stacking with hydrogen of protein aromatic center
                            else:
                                if ai_distance >= 4.6:
                                    # check offset between grid point and aromatic center
                                    offset = adjacent(alpha, ai_distance)
                                    if 0.001 <= offset <= 0.5:
                                        ai_n2 = cross_product(ai_n, cross_product(ai_n, ai_vector))
                                        ai_n2, alpha = ai_geometry(ai_vector, ai_n2)
                                        grid_score['ai'][ind] += t_stacking_distance_score_dict[round(ai_distance, 1)]
                                        grid_partners[ind][grid_list_dict['ai_i']] += t_stacking_partner_position(
                                                                                      ai_i, grid_point, ai_n2,
                                                                                      offset, ai_distance, alpha)
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
        logger.debug('Trajectory {} finished with frame {}.'.format(counter + 1, frame + 1))
    logger.info('Finished analysis of trajectory {}.'.format(counter + 1))
    grid_partners = grid_partners_to_array(grid_partners)
    return [grid_score, grid_partners]
