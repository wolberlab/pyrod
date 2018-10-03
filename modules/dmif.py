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
    from pyrod.modules.lookup import grid_list_dict, hb_dist_dict, hb_angl_dict, hd_sel_dict, sel_cutoff_dict, \
        pi_stacking_distance_score_dict, t_stacking_distance_score_dict, ai_pi_distance_score_dict, \
        AI_PI_ANGLE_CUTOFF
    from pyrod.modules.helper_dmif import main_selection, hd_selection, ha_selection, hi_selection, ni_selection, \
        pi_selection, ai_selection, metal_selection, buriedness, pi_stacking_partner_position, grid_parameters, \
        grid_partners_to_array, ai_geometry, t_stacking_partner_position
    from pyrod.modules.helper_math import distance, angle, normal, opposite, adjacent, norm, vector_angle, vector, \
        cross_product
    from pyrod.modules.helper_update import update_progress_dmif_parameters, update_progress_dmif
    from pyrod.modules.helper_write import setup_logger
except ImportError:
    from modules.lookup import grid_list_dict, hb_dist_dict, hb_angl_dict, hd_sel_dict, sel_cutoff_dict, \
        pi_stacking_distance_score_dict, t_stacking_distance_score_dict, ai_pi_distance_score_dict, AI_PI_ANGLE_CUTOFF
    from modules.helper_dmif import main_selection, hd_selection, ha_selection, hi_selection, ni_selection, \
        pi_selection, ai_selection, metal_selection, buriedness, pi_stacking_partner_position, grid_parameters, \
        grid_partners_to_array, ai_geometry, t_stacking_partner_position
    from modules.helper_math import distance, angle, normal, opposite, adjacent, norm, vector_angle, vector, \
        cross_product
    from modules.helper_update import update_progress_dmif_parameters, update_progress_dmif
    from modules.helper_write import setup_logger


def dmif(topology, trajectory, counter, length_trajectory, number_processes, number_trajectories, grid_score,
         grid_partners, first_frame, last_frame, metal_names, directory, debugging, get_partners):
    logger = setup_logger('_'.join(['dmif_trajectory', str(counter + 1)]), directory, debugging)
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
                             u.atoms.types)], dtype=[('atomid', 'i4'), ('resname', 'U10'), ('resid', 'i4'),
                                                     ('name', 'U10'), ('type', 'U10')])
    positions = np.array([[x, y, z] for x, y, z in zip(grid_score['x'], grid_score['y'], grid_score['z'])])
    x_minimum, x_maximum, y_minimum, y_maximum, z_minimum, z_maximum = grid_parameters(positions)[:-1]
    tree = cKDTree(positions)
    main_atoms = main_selection(topology)
    hd_atomids, hd_types, hd_hydrogen_atomid_lists = hd_selection(main_atoms)
    ha_atomids, ha_types = ha_selection(main_atoms)
    hi_atomids = hi_selection(main_atoms)
    ni_atomids = ni_selection(main_atoms)
    pi_atomids = pi_selection(main_atoms)
    ai_atomids = ai_selection(main_atoms)
    metal_atomids = metal_selection(topology, metal_names)
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
        h2os_os_box_inds = topology[((topology['resname'] == 'HOH') & (topology['name'] == 'O') &
                                    (positions[:, 0] >= x_minimum) & (positions[:, 0] <= x_maximum) &
                                    (positions[:, 1] >= y_minimum) & (positions[:, 1] <= y_maximum) &
                                    (positions[:, 2] >= z_minimum) & (positions[:, 2] <= z_maximum))]['atomid']
        if len(h2os_os_box_inds) > 0:
            tree_h2os = cKDTree(positions[h2os_os_box_inds])
            if len(hd_atomids) > 0:
                hd_positions = positions[hd_atomids]
                hd_lists = tree_h2os.query_ball_tree(cKDTree(hd_positions), sel_cutoff_dict['hb'])
            else:
                hd_positions = []
                hd_lists = [[]] * len(h2os_os_box_inds)
            if len(ha_atomids) > 0:
                ha_positions = positions[ha_atomids]
                ha_lists = tree_h2os.query_ball_tree(cKDTree(ha_positions), sel_cutoff_dict['hb'])
            else:
                ha_positions = []
                ha_lists = [[]] * len(h2os_os_box_inds)
            if len(hi_atomids) > 0:
                hi_positions = positions[hi_atomids]
                hi_lists = tree_h2os.query_ball_tree(cKDTree(hi_positions), sel_cutoff_dict['hi'])
            else:
                hi_positions = []
                hi_lists = [[]] * len(h2os_os_box_inds)
            if len(ni_atomids) > 0:
                ni_positions = [((x + y) / 2) for x, y in zip(positions[ni_atomids[::2]],
                                                              positions[ni_atomids[1::2]])]
                ni_lists = tree_h2os.query_ball_tree(cKDTree(ni_positions), sel_cutoff_dict['ii'])
            else:
                ni_positions = []
                ni_lists = [[]] * len(h2os_os_box_inds)
            if len(pi_atomids) > 0:
                pi_positions = [((x + y) / 2) for x, y in zip(positions[pi_atomids[::2]],
                                                              positions[pi_atomids[1::2]])]
                pi_lists = tree_h2os.query_ball_tree(cKDTree(pi_positions), sel_cutoff_dict['ii'])
            else:
                pi_positions = []
                pi_lists = [[]] * len(h2os_os_box_inds)
            if len(ai_atomids) > 0:
                ai_positions = [((x + y + z) / 3) for x, y, z in zip(positions[ai_atomids[::3]],
                                                                     positions[ai_atomids[1::3]],
                                                                     positions[ai_atomids[2::3]])]
                ai_normals = [normal(a, b, c) for a, b, c in zip(positions[ai_atomids[::3]], ai_positions,
                                                                 positions[ai_atomids[2::3]])]
                ai_lists = tree_h2os.query_ball_tree(cKDTree(ai_positions), sel_cutoff_dict['ai'])
            else:
                ai_positions = []
                ai_normals = []
                ai_lists = [[]] * len(h2os_os_box_inds)
            if len(metal_atomids) > 0:
                metal_positions = positions[metal_atomids]
                metal_lists = tree_h2os.query_ball_tree(cKDTree(metal_positions), sel_cutoff_dict['metal'])
            else:
                metal_positions = []
                metal_lists = [[]] * len(h2os_os_box_inds)
        else:
            h2os_os_box_inds = []
            hd_positions, ha_positions, hi_positions, ni_positions = [], [], [], []
            pi_positions, ai_positions, ai_normals, metal_positions = [], [], [], []
            hd_lists, ha_lists, hi_lists, ni_lists, pi_lists, ai_lists, metal_lists = [], [], [], [], [], [], []
        for o_ind, hd_list, ha_list, hi_list, ni_list, pi_list, ai_list, metal_list in \
                zip(h2os_os_box_inds, hd_lists, ha_lists, hi_lists, ni_lists, pi_lists, ai_lists, metal_lists):
            ha, ha_i, hd, hd_i, hi, pi, ni, ai, ai_i, ai_n = 0, [], 0, [], 0, 0, 0, 0, [], []
            o_coor, h1_coor, h2_coor = positions[o_ind], positions[o_ind + 1], positions[o_ind + 2]
            # hydrogen bond acceptor feature
            for hd_ind in hd_list:
                hd_coor, hd_type, hd_hydrogen_coors = [hd_positions[hd_ind], hd_types[hd_ind],
                                                       positions[hd_hydrogen_atomid_lists[hd_ind]]]
                if distance(o_coor, hd_coor) <= hb_dist_dict[hd_type]:
                    for hd_hydrogen_coor in hd_hydrogen_coors:
                        if angle(o_coor, hd_hydrogen_coor, hd_coor) >= hb_angl_dict[hd_type]:
                            ha += 1
                            ha_i.append(hd_coor)
            # hydrogen bond donor feature
            for ha_ind in ha_list:
                ha_coor, ha_type = ha_positions[ha_ind], ha_types[ha_ind]
                if distance(o_coor, ha_coor) <= hb_dist_dict[ha_type]:
                    for h_coor in [h1_coor, h2_coor]:
                        if angle(ha_coor, h_coor, o_coor) >= hb_angl_dict[ha_type]:
                            hd += 1
                            hd_i.append(ha_coor)
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
                        if get_partners:
                            for ind in inds:
                                grid_partners[ind][grid_list_dict['ha_i']] += ha_i
                    # double
                    elif ha == 2:
                        ha2_inds += inds
                        if get_partners:
                            for ind in inds:
                                grid_partners[ind][grid_list_dict['ha2_i']] += ha_i
                # single hydrogen bond donors
                elif hd == 1:
                    # single donor
                    if ha == 0:
                        hd_inds += inds
                        if get_partners:
                            for ind in inds:
                                grid_partners[ind][grid_list_dict['hd_i']] += hd_i
                    # mixed donor acceptor
                    elif ha == 1:
                        hda_inds += inds
                        if get_partners:
                            for ind in inds:
                                grid_partners[ind][grid_list_dict['hda_id']] += hd_i
                                grid_partners[ind][grid_list_dict['hda_ia']] += ha_i
                else:
                    # double hydrogen bond donor
                    hd2_inds += inds
                    if get_partners:
                        for ind in inds:
                            grid_partners[ind][grid_list_dict['hd2_i']] += hd_i
                # ionizable interactions
                # sum up negative ionizable
                for pi_ind in pi_list:
                    ni += 2.6 / distance(o_coor, pi_positions[pi_ind])
                # sum up positive ionizable
                for ni_ind in ni_list:
                    pi += 2.6 / distance(o_coor, ni_positions[ni_ind])
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
                    grid_score['hi_norm'][inds] += hi
                    # no charged amino acid within 4 A
                    if ni < 0.65 > pi:
                        grid_score['hi'][inds] += hi
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
                                        if get_partners:
                                            grid_partners[ind][grid_list_dict['ai_i']] += \
                                                pi_stacking_partner_position(grid_point, ai_n, ai_distance, alpha)
                                # t-stacking
                                else:
                                    # check offset between grid point and aromatic center
                                    if 0.001 <= offset <= 0.5:
                                        grid_score['ai'][ind] += t_stacking_distance_score_dict[round(ai_distance, 1)]
                                        if get_partners:
                                            grid_partners[ind][grid_list_dict['ai_i']] += \
                                                t_stacking_partner_position(ai_i, grid_point, ai_n, offset,
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
                                        if get_partners:
                                            grid_partners[ind][grid_list_dict['ai_i']] += \
                                                t_stacking_partner_position(ai_i, grid_point, ai_n2, offset,
                                                                            ai_distance, alpha)
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
