""" PyRod - dynamic molecular interaction fields (dMIFs), based on tracing water molecules in MD simulations.

This module analyzes molecular dynamics simulations and stores the results in arrays.
"""

# python standard libraries
import os
import sys
import time
import warnings

# external libraries
import math
import MDAnalysis as mda
import MDAnalysis.analysis.encore as encore
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

# pyrod modules
try:
    from pyrod.modules.lookup import grid_list_dict, hb_dist_dict, hb_angl_dict, hd_sel_dict, sel_cutoff_dict, \
        pi_stacking_distance_score_dict, t_stacking_distance_score_dict, cation_pi_distance_score_dict, \
        CATION_PI_ANGLE_CUTOFF, standard_resnames_dict, CLASH_CUTOFF
    from pyrod.modules.helper_read import translate_pharmacophore
    from pyrod.modules.helper_trajectory import main_selection, hd_selection, ha_selection, hi_selection, \
        ni_selection, pi_selection, ai_selection, metal_selection, buriedness, pi_stacking_partner_position, \
        grid_parameters, grid_partners_to_array, ai_geometry, t_stacking_partner_position
    from pyrod.modules.helper_math import distance, angle, normal, opposite, adjacent, norm, vector_angle, vector, \
        cross_product
    from pyrod.modules.helper_update import update_progress_dmif_parameters, update_progress_dmif, update_user
    from pyrod.modules.helper_write import setup_logger, file_path
except ImportError:
    from modules.lookup import grid_list_dict, hb_dist_dict, hb_angl_dict, hd_sel_dict, sel_cutoff_dict, \
        pi_stacking_distance_score_dict, t_stacking_distance_score_dict, cation_pi_distance_score_dict, \
        CATION_PI_ANGLE_CUTOFF, standard_resnames_dict, CLASH_CUTOFF
    from modules.helper_read import translate_pharmacophore
    from modules.helper_trajectory import main_selection, hd_selection, ha_selection, hi_selection, ni_selection, \
        pi_selection, ai_selection, metal_selection, buriedness, pi_stacking_partner_position, grid_parameters, \
        grid_partners_to_array, ai_geometry, t_stacking_partner_position
    from modules.helper_math import distance, angle, normal, opposite, adjacent, norm, vector_angle, vector, \
        cross_product
    from modules.helper_update import update_progress_dmif_parameters, update_progress_dmif, update_user
    from modules.helper_write import setup_logger, file_path


def trajectory_analysis(topology, trajectory, counter, length_trajectory, number_processes, number_trajectories,
                        grid_score, grid_partners, first_frame, last_frame, metal_names, directory, debugging,
                        get_partners):
    logger = setup_logger('_'.join(['dmif_trajectory', str(counter)]), directory, debugging)
    logger.info('Started analysis of trajectory {}.'.format(counter))
    check_progress, final, past_frames, future_frames = update_progress_dmif_parameters(
        counter, length_trajectory, number_processes, number_trajectories)
    if debugging:
        u = mda.Universe(topology, trajectory)
    else:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            u = mda.Universe(topology, trajectory)
    dtype = [('atomid', int), ('resname', 'U10'), ('resid', int), ('name', 'U10'), ('type', 'U10')]
    topology = np.array([(a, b, c, d, e) for a, b, c, d, e in
                         zip(range(len(u.atoms.resnames)), u.atoms.resnames, u.atoms.resids, u.atoms.names,
                         u.atoms.types)], dtype=dtype)
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
        h2o_inds = []
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
                            ha_i.append([float(x) for x in hd_coor])
            # hydrogen bond donor feature
            for ha_ind in ha_list:
                ha_coor, ha_type = ha_positions[ha_ind], ha_types[ha_ind]
                if distance(o_coor, ha_coor) <= hb_dist_dict[ha_type]:
                    for h_coor in [h1_coor, h2_coor]:
                        if angle(ha_coor, h_coor, o_coor) >= hb_angl_dict[ha_type]:
                            hd += 1
                            hd_i.append([float(x) for x in ha_coor])
            # metals
            for metal_ind in metal_list:
                metal_position = metal_positions[metal_ind]
                ha += 1
                ha_i.append([float(x) for x in metal_position])
                ni += 2.6 / distance(o_coor, metal_position)
            # indices of points close to water
            inds = tree.query_ball_point(o_coor, r=1.41)
            h2o_inds += inds
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
                                grid_partners[ind][grid_list_dict['ha']] += ha_i
                    # double
                    elif ha == 2:
                        ha2_inds += inds
                        if get_partners:
                            for ind in inds:
                                grid_partners[ind][grid_list_dict['ha2']] += ha_i
                # single hydrogen bond donors
                elif hd == 1:
                    # single donor
                    if ha == 0:
                        hd_inds += inds
                        if get_partners:
                            for ind in inds:
                                grid_partners[ind][grid_list_dict['hd']] += hd_i
                    # mixed donor acceptor
                    elif ha == 1:
                        hda_inds += inds
                        if get_partners:
                            for ind in inds:
                                grid_partners[ind][grid_list_dict['hda']][0] += hd_i
                                grid_partners[ind][grid_list_dict['hda']][1] += ha_i
                else:
                    # double hydrogen bond donor
                    hd2_inds += inds
                    if get_partners:
                        for ind in inds:
                            grid_partners[ind][grid_list_dict['hd2']] += hd_i
                # ionizable interactions and cation-pi interactions
                # negative ionizable and cation-pi interactions
                for pi_ind in pi_list:
                    pi_i = pi_positions[pi_ind]
                    # negative ionizable interaction
                    ni += 2.6 / distance(o_coor, pi_i)
                    # cation-pi interaction
                    for ind in inds:
                        grid_point = [grid_score['x'][ind], grid_score['y'][ind], grid_score['z'][ind]]
                        pi_distance = distance(grid_point, pi_i)
                        if 3.1 <= pi_distance <= 6.0:
                            grid_score['ai'][ind] += cation_pi_distance_score_dict[round(pi_distance, 1)]
                            if get_partners:
                                grid_partners[ind][grid_list_dict['ai']] += [[float(x) for x in pi_i]]
                # positive ionizable
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
                    # no charged environment
                    if ni < 0.65 > pi:
                        grid_score['hi'][inds] += hi
                # aromatic interactions grid point wise
                for ai_ind in ai_list:
                    ai_i = ai_positions[ai_ind]
                    ai_n = ai_normals[ai_ind]
                    for ind in inds:
                        grid_point = [grid_score['x'][ind], grid_score['y'][ind], grid_score['z'][ind]]
                        ai_distance = distance(grid_point, ai_i)
                        if 3.1 <= ai_distance <= 6.0:
                            ai_vector = vector(ai_i, grid_point)
                            ai_n, alpha = ai_geometry(ai_vector, ai_n)
                            # cation-pi interactions
                            if alpha <= CATION_PI_ANGLE_CUTOFF:
                                grid_score['pi'][ind] += cation_pi_distance_score_dict[round(ai_distance, 1)]
                            # pi- and t-stacking
                            if ai_distance >= 3.3:
                                # pi- and t-stacking with pi-system of protein aromatic center
                                if alpha < 45:
                                    offset = opposite(alpha, ai_distance)
                                    # pi-stacking
                                    if ai_distance <= 4.7:
                                        # check offset between grid point and aromatic center
                                        if offset <= 2.0:
                                            grid_score['ai'][ind] += pi_stacking_distance_score_dict[round(ai_distance,
                                                                                                           1)]
                                            if get_partners:
                                                grid_partners[ind][grid_list_dict['ai']] += \
                                                    pi_stacking_partner_position(grid_point, ai_n, ai_distance, alpha)
                                    # t-stacking
                                    else:
                                        # check offset between grid point and aromatic center
                                        if offset <= 0.5:
                                            grid_score['ai'][ind] += t_stacking_distance_score_dict[round(ai_distance,
                                                                                                          1)]
                                            if get_partners:
                                                grid_partners[ind][grid_list_dict['ai']] += \
                                                    t_stacking_partner_position(ai_i, grid_point, ai_n, offset,
                                                                                ai_distance, alpha, True)
                                # t-stacking with hydrogen of protein aromatic center
                                else:
                                    if ai_distance >= 4.6:
                                        # check offset between grid point and aromatic center
                                        offset = adjacent(alpha, ai_distance)
                                        if offset <= 0.5:
                                            grid_score['ai'][ind] += t_stacking_distance_score_dict[round(ai_distance,
                                                                                                          1)]
                                            if get_partners:
                                                ai_n2 = cross_product(ai_n, cross_product(ai_n, ai_vector))
                                                ai_n2, alpha = ai_geometry(ai_vector, ai_n2)
                                                grid_partners[ind][grid_list_dict['ai']] += \
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
        grid_score['h2o'][h2o_inds] += 1
        if check_progress:
            update_progress_dmif(counter, frame, length_trajectory, number_trajectories, number_processes, past_frames,
                                 future_frames, start, final)
        logger.debug('Trajectory {} finished with frame {}.'.format(counter, frame))
    logger.info('Finished analysis of trajectory {}.'.format(counter))
    grid_partners = grid_partners_to_array(grid_partners)
    return [grid_score, grid_partners]


def screen_protein_conformations(topology, trajectory, pharmacophore_path, ligand_path, counter, first_frame,
                                 last_frame, metal_names, directory, output_name, debugging):
    dcd_name = 'ensemble_' + str(counter) + '.dcd'
    output_directory = '/'.join([directory, output_name])
    file_path(dcd_name, output_directory)
    logger = setup_logger('_'.join(['screen_protein_conformations', str(counter)]), directory, debugging)
    logger.info('Started screening of protein conformations in trajectory {}.'.format(counter))
    ligand_positions = None
    if debugging:
        u = mda.Universe(topology, trajectory)
        if ligand_path:
            ligand_positions = mda.Universe(ligand_path).atoms.positions
    else:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            u = mda.Universe(topology, trajectory)
            if ligand_path:
                ligand_positions = mda.Universe(ligand_path).atoms.positions
    protein = u.select_atoms('protein')
    dtype = [('atomid', int), ('resname', 'U10'), ('resid', int), ('name', 'U10'), ('type', 'U10')]
    topology = np.array([(a, b, c, d, e) for a, b, c, d, e in
                         zip(range(len(u.atoms.resnames)), u.atoms.resnames, u.atoms.resids, u.atoms.names,
                         u.atoms.types)], dtype=dtype)
    main_atoms = main_selection(topology)
    main_atomids = main_atoms['atomid']
    hd_atomids = hd_selection(main_atoms)[0]
    ha_atomids = ha_selection(main_atoms)[0]
    hi_atomids = hi_selection(main_atoms)
    ni_atomids = ni_selection(main_atoms)
    pi_atomids = pi_selection(main_atoms)
    ai_atomids = ai_selection(main_atoms)
    metal_atomids = metal_selection(topology, metal_names)
    features = translate_pharmacophore(pharmacophore_path)
    if counter == 0:
        file_path('protein.pdb', output_directory)
        with mda.Writer('/'.join([output_directory, 'protein.pdb']), bonds=None, n_atoms=protein.n_atoms) as PDB:
            PDB.write(protein)
    frame_collector = []
    with mda.Writer('/'.join([output_directory, dcd_name]), n_atoms=protein.n_atoms) as DCD:
        for frame, _ in enumerate(u.trajectory[first_frame:last_frame:]):
            positions = u.atoms.positions
            matched_features = 0
            for feature in features:
                ai, ha, hd, pi, ni, cation_pi, hi = 0, 0, 0, 0, 0, 0, 0
                feature_type = feature['type']
                feature_position = np.array(list(feature[['x', 'y', 'z']]))
                feature_position_i = np.array(list(feature[['xi', 'yi', 'zi']]))
                feature_tolerance_i = feature['tolerance_i']
                # hydrogen bonds and metal interaction
                if feature_type == 'hd':
                    if len(ha_atomids) > 0:
                        ha_positions = positions[ha_atomids]
                        ha += np.sum((cdist(feature_position_i.reshape(1, 3), ha_positions) <= feature_tolerance_i)[0])
                    if ha == 0:
                        break
                    else:
                        matched_features += 1
                elif feature_type == 'ha':
                    if len(hd_atomids) > 0:
                        hd_positions = positions[hd_atomids]
                        hd += np.sum((cdist(feature_position_i.reshape(1, 3), hd_positions) <= feature_tolerance_i)[0])
                    if len(metal_atomids) > 0:
                        metal_positions = positions[metal_atomids]
                        hd += np.sum((cdist(feature_position_i.reshape(1, 3), metal_positions) <=
                                      feature_tolerance_i)[0])
                    if hd == 0:
                        break
                    else:
                        matched_features += 1
                elif feature_type in ['hi', 'pi', 'ni']:
                    if len(ni_atomids) > 0:
                        ni_positions = np.array([((x + y) / 2) for x, y in zip(positions[ni_atomids[::2]],
                                                                               positions[ni_atomids[1::2]])])
                        ni_positions = ni_positions[(cdist(feature_position.reshape(1, 3), ni_positions) <=
                                                     sel_cutoff_dict['ii'])[0]]
                        for ni_position in ni_positions:
                            pi += 2.6 / distance(feature_position, ni_position)
                    if len(pi_atomids) > 0:
                        pi_positions = np.array([((x + y) / 2) for x, y in zip(positions[pi_atomids[::2]],
                                                                               positions[pi_atomids[1::2]])])
                        pi_positions = pi_positions[(cdist(feature_position.reshape(1, 3), pi_positions) <=
                                                     sel_cutoff_dict['ii'])[0]]
                        for pi_position in pi_positions:
                            ni += 2.6 / distance(feature_position, pi_position)
                    if len(metal_atomids) > 0:
                        metal_positions = positions[metal_atomids]
                        metal_booleans = (cdist(feature_position.reshape(1, 3), metal_positions) <=
                                          sel_cutoff_dict['metal'])[0]
                        metal_positions = metal_positions[metal_booleans]
                        for metal_position in metal_positions:
                            ni += 2.6 / distance(feature_position, metal_position)
                    if feature_type == 'hi':
                        if len(hi_atomids) > 0:
                            # no charged protein environment
                            if ni >= 0.65 <= pi:
                                break
                            hi_positions = positions[hi_atomids]
                            hi_positions = hi_positions[(cdist(feature_position.reshape(1, 3), hi_positions) <=
                                                         sel_cutoff_dict['hi'])[0]]
                            if len(hi_positions) > 0:
                                hi += 1
                                if len(hi_positions) > 1:
                                    hi += buriedness(feature_position, hi_positions)
                            # check if pi and ni > 0.65
                            if hi < feature['score']:
                                break
                            else:
                                matched_features += 1
                    elif feature_type == 'pi':
                        if len(ai_atomids) > 0:
                            # cation-pi interactions
                            ai_positions = np.array([((x + y + z) / 3) for x, y, z in zip(positions[ai_atomids[::3]],
                                                                                          positions[ai_atomids[1::3]],
                                                                                          positions[ai_atomids[2::3]])])
                            ai_normals = np.array([normal(a, b, c) for a, b, c in zip(positions[ai_atomids[::3]],
                                                                                      ai_positions,
                                                                                      positions[ai_atomids[2::3]])])
                            ai_booleans = (cdist(feature_position.reshape(1, 3), ai_positions) <=
                                           sel_cutoff_dict['ai'])[0]
                            ai_positions = ai_positions[ai_booleans]
                            ai_normals = ai_normals[ai_booleans]
                            for ai_i, ai_n in zip(ai_positions, ai_normals):
                                ai_distance = distance(ai_i, feature_position)
                                if 3.1 <= ai_distance <= 6.0:
                                    ai_n, alpha = ai_geometry(vector(ai_i, feature_position), ai_n)
                                    if alpha <= CATION_PI_ANGLE_CUTOFF:
                                        cation_pi += cation_pi_distance_score_dict[round(ai_distance, 1)]
                        if pi + cation_pi - ni < feature['score']:
                            break
                        else:
                            matched_features += 1
                    elif feature_type == 'ni':
                        if ni - pi < feature['score']:
                            break
                        else:
                            matched_features += 1
                elif feature_type == 'ai':
                    if len(pi_atomids) > 0:
                        # cation-pi interaction
                        pi_positions = np.array([((x + y) / 2) for x, y in zip(positions[pi_atomids[::2]],
                                                                               positions[pi_atomids[1::2]])])
                        pi_positions = pi_positions[(cdist(feature_position.reshape(1, 3), pi_positions) <=
                                                     sel_cutoff_dict['ii'])[0]]
                        for pi_position in pi_positions:
                            pi_distance = distance(pi_position, feature_position)
                            if 3.1 <= pi_distance <= 6.0:
                                alpha = ai_geometry(vector(pi_position, feature_position),
                                                    feature[['xi', 'yi', 'zi']])[1]
                                if alpha <= CATION_PI_ANGLE_CUTOFF:
                                    ai += cation_pi_distance_score_dict[round(pi_distance, 1)]
                    if len(ai_atomids) > 0:
                        # aromatic interactions
                        ai_positions = np.array([((x + y + z) / 3) for x, y, z in zip(positions[ai_atomids[::3]],
                                                                                      positions[ai_atomids[1::3]],
                                                                                      positions[ai_atomids[2::3]])])
                        ai_normals = np.array([normal(a, b, c) for a, b, c in zip(positions[ai_atomids[::3]],
                                                                                  ai_positions,
                                                                                  positions[ai_atomids[2::3]])])
                        ai_booleans = (cdist(feature_position.reshape(1, 3), ai_positions) <= sel_cutoff_dict['ai'])[0]
                        ai_positions = ai_positions[ai_booleans]
                        ai_normals = ai_normals[ai_booleans]
                        for ai_i, ai_n in zip(ai_positions, ai_normals):
                            ai_distance = distance(ai_i, feature_position)
                            if 3.3 <= ai_distance <= 6.0:
                                ai_vector = vector(ai_i, feature_position)
                                ai_n, alpha = ai_geometry(ai_vector, ai_n)
                                angle_tolerance = math.degrees(feature_tolerance_i)
                                # pi- and t-stacking with pi-system of protein aromatic center
                                if alpha < 45:
                                    offset = opposite(alpha, ai_distance)
                                    # pi-stacking
                                    if ai_distance <= 4.7:
                                        # check offset between grid point and aromatic center
                                        if offset <= 2.0:
                                            # check angle between normals
                                            if vector_angle(ai_n, feature[['xi', 'yi', 'zi']]) <= angle_tolerance:
                                                ai += pi_stacking_distance_score_dict[round(ai_distance, 1)]
                                    # t-stacking
                                    else:
                                        # check offset between grid point and aromatic center
                                        if offset <= 0.5:
                                            # check angle between normals
                                            if (90 - angle_tolerance <= vector_angle(ai_n, feature[['xi', 'yi', 'zi']])
                                                    >= 90 + angle_tolerance):
                                                ai += t_stacking_distance_score_dict[round(ai_distance, 1)]
                                # t-stacking with hydrogen of protein aromatic center
                                else:
                                    if ai_distance >= 4.6:
                                        offset = adjacent(alpha, ai_distance)
                                        # check offset between grid point and aromatic center
                                        if offset <= 0.5:
                                            if (90 - angle_tolerance <= vector_angle(ai_n, feature[['xi', 'yi', 'zi']])
                                                    >= 90 + angle_tolerance):
                                                ai += t_stacking_distance_score_dict[round(ai_distance, 1)]
                    if ai < feature['score']:
                        break
                    else:
                        matched_features += 1
            if matched_features == len(features):
                clash = False
                main_positions = positions[main_atomids]
                if cdist(main_positions, np.array([list(feature[['x', 'y', 'z']]) for feature in
                                                   features])).min() < CLASH_CUTOFF:
                    clash = True
                if ligand_path:
                    if cdist(main_positions, ligand_positions).min() < CLASH_CUTOFF:
                        clash = True
                if not clash:
                    DCD.write(protein)
                    frame_collector.append(frame + first_frame)
            logger.debug('Trajectory {} finished with frame {}.'.format(counter, frame))
    logger.info('Finished screening of trajectory {}.'.format(counter))
    with open('{}/frames_{}.csv'.format(output_directory, counter), 'w') as csv:
        for frame in frame_collector:
            csv.write('{}\t{}\n'.format(counter, frame))
    return


def ensembles_to_centroid(topology, trajectories, output_name, directory, debugging):
    logger = setup_logger('ensembles_to_centroid', directory, debugging)
    output_directory = '/'.join([directory, output_name])
    protein_topology = '/'.join([output_directory, 'protein.pdb'])
    protein_trajectories = []
    frames = []
    # check if frames in trajectory files, delete empty trajectory files, collect frames in list
    for x in range(len(trajectories)):
        with open('{}/frames_{}.csv'.format(output_directory, x), 'r') as csv:
            frames += csv.readlines()
        os.remove('{}/frames_{}.csv'.format(output_directory, x))
        try:
            mda.Universe(protein_topology, '/'.join([output_directory, 'ensemble_{}.dcd'.format(x)]))
            protein_trajectories.append(x)
        except OSError:
            os.remove('/'.join([output_directory, 'ensemble_{}.dcd'.format(x)]))
    if len(protein_trajectories) > 0:
        # info to user
        if len(frames) > 1:
            update_user('Getting centroid from {} protein conformations.'.format(len(frames)), logger)
        else:
            update_user('Found only 1 protein conformations.', logger)
        # merge trajectories into one file
        protein_trajectories = ['/'.join([output_directory, 'ensemble_{}.dcd'.format(x)]) for x in protein_trajectories]
        u = mda.Universe(protein_topology, protein_trajectories)
        with mda.Writer('/'.join([output_directory, 'ensemble.dcd']), n_atoms=u.atoms.n_atoms) as DCD:
            for _ in u.trajectory:
                DCD.write(u.atoms)
        # remove sub trajectories
            for path in protein_trajectories:
                os.remove(path)
        # find centroid of frames
        if len(frames) > 1:
            u = mda.Universe(protein_topology, '/'.join([output_directory, 'ensemble.dcd']))
            conf_dist_matrix = encore.confdistmatrix.get_distance_matrix(u, selection='all', superimpose=True, n_job=1,
                                                                         weights='mass', metadata=False, verbose=False)
            centroid = conf_dist_matrix.as_array().sum(axis=1).argmin()
        else:
            centroid = 0
        # write centroid
        u = mda.Universe(topology, trajectories[int(frames[centroid].split()[0])])
        file_path('centroid.pdb', output_directory)
        with mda.Writer('/'.join([output_directory, 'centroid.pdb']), bonds=None, n_atoms=u.atoms.n_atoms) as PDB:
            for _ in u.trajectory[int(frames[centroid].split()[1]):int(frames[centroid].split()[1]) + 1:]:
                PDB.write(u.atoms)
        # write csv with frame references
        file_path('frames.csv', output_directory)
        frames[centroid] = '{}\t{}\t{}\n'.format(frames[centroid].split()[0], frames[centroid].split()[1], 'centroid')
        with open('{}/frames.csv'.format(output_directory), 'w') as csv:
            csv.write(''.join(['trajectory\tframe\tcentroid\n'] + frames))
    else:
        update_user('No protein conformations found.', logger)
        sys.exit()
    return
