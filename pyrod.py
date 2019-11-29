""" PyRod - dynamic molecular interaction fields (dMIFs), based on tracing water molecules in MD simulations.

Released under the GNU Public Licence v2.

This is the main script to run PyRod from the command line for analyzing molecular dynamics simulations and generating
dMIFs, pharmacophores and centroids.
"""

# python standard libraries
import argparse
import configparser
import multiprocessing
import os
import time
import warnings

# external libraries
import numpy as np

# pyrod modules
try:
    from pyrod.pyrod_lib.grid import generate_grid, dmif_data_structure, post_processing, generate_dmif_excess, \
        get_point_properties
    from pyrod.pyrod_lib.config import test_grid_parameters, trajectory_analysis_parameters, \
        exclusion_volume_parameters, feature_parameters, pharmacophore_parameters, library_parameters, \
        dmif_excess_parameters, centroid_parameters, point_properties_parameters
    from pyrod.pyrod_lib.lookup import logo, __version__, grid_list_dict, feature_types
    from pyrod.pyrod_lib.pharmacophore import generate_exclusion_volumes, generate_features, generate_library
    from pyrod.pyrod_lib.pharmacophore_helper import renumber_features
    from pyrod.pyrod_lib.read import pickle_reader
    from pyrod.pyrod_lib.trajectory import trajectory_analysis, screen_protein_conformations, ensemble_to_centroid
    from pyrod.pyrod_lib.write import file_path, pdb_grid, dmif_writer, pharmacophore_writer, pickle_writer, \
        setup_logger, update_user, time_to_text
except ImportError:
    from pyrod_lib.grid import generate_grid, dmif_data_structure, post_processing, generate_dmif_excess, \
        get_point_properties
    from pyrod_lib.config import test_grid_parameters, trajectory_analysis_parameters, \
        exclusion_volume_parameters, feature_parameters, pharmacophore_parameters, library_parameters, \
        dmif_excess_parameters, centroid_parameters, point_properties_parameters
    from pyrod_lib.lookup import logo, __version__, grid_list_dict, feature_types
    from pyrod_lib.pharmacophore import generate_exclusion_volumes, generate_features, generate_library
    from pyrod_lib.pharmacophore_helper import renumber_features
    from pyrod_lib.read import pickle_reader
    from pyrod_lib.trajectory import trajectory_analysis, screen_protein_conformations, ensemble_to_centroid
    from pyrod_lib.write import file_path, pdb_grid, dmif_writer, pharmacophore_writer, pickle_writer, \
        setup_logger, time_to_text, update_user


def chunks(iterable, chunk_size):
    """ This functions returns a list of chunks. """
    for i in range(0, len(iterable), chunk_size):
        yield iterable[i:i + chunk_size]


if __name__ == '__main__':
    start_time = time.time()
    parser = argparse.ArgumentParser(prog='PyRod', description='\n'.join(logo),
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('conf', help='path to configuration file')
    parser.add_argument('--verbose', dest='debugging', action='store_true', help='verbose logging for debugging')
    conf = parser.parse_args().conf
    debugging = parser.parse_args().debugging
    config = configparser.ConfigParser()
    config.read(conf)
    directory = config.get('directory', 'directory')
    if len(directory) == 0:
        directory = os.getcwd() + '/pyrod'
    logger = setup_logger('main', directory, debugging)
    update_user('\n'.join(logo), logger)
    logger.debug('\n'.join([': '.join(list(_)) for _ in config.items('directory')]))
    # defining grid
    if config.has_section('test grid parameters'):
        logger.debug('\n'.join([': '.join(list(_)) for _ in config.items('test grid parameters')]))
        center, edge_lengths, name = test_grid_parameters(config)
        # determine space resulting in less than 100000 grid points
        space = 0.5
        space_found = False
        while not space_found:
            grid = generate_grid(center, edge_lengths, space)
            if len(grid) < 100000:
                space_found = True
                pdb_grid(grid, name, '{}/test'.format(directory))
                update_user('Writing test grid to {}/test.'.format(directory), logger)
            else:
                if space == 0.5:
                    space += 0.5
                else:
                    space += 1
    # point properties
    if config.has_section('point properties parameters'):
        logger.debug('\n'.join([': '.join(list(_)) for _ in config.items('point properties parameters')]))
        point, dmif_path = point_properties_parameters(config)
        update_user('Getting point properties from {}.'.format(dmif_path), logger)
        point_properties_dict = get_point_properties(point, dmif_path)
        for key, value in point_properties_dict.items():
            update_user('{}: {}'.format(key, value), logger)
    # trajectory analysis
    if config.has_section('trajectory analysis parameters'):
        update_user('Starting trajectory analysis.', logger)
        logger.debug('\n'.join([': '.join(list(_)) for _ in config.items('trajectory analysis parameters')]))
        center, edge_lengths, topology, trajectories, first_frame, last_frame, step_size, total_number_of_frames, \
            metal_names, map_formats, number_of_processes, get_partners = trajectory_analysis_parameters(config,
                                                                                                         debugging)
        update_user('Initializing grid.', logger)
        grid_score, grid_partners = dmif_data_structure(generate_grid(center, edge_lengths), get_partners)
        manager = multiprocessing.Manager()
        results = manager.list()
        frame_counter = multiprocessing.Value('i', 0)
        trajectory_time = time.time()
        processes = [multiprocessing.Process(target=trajectory_analysis, args=(topology, trajectory, grid_score,
                     grid_partners, frame_counter, total_number_of_frames, first_frame, last_frame, step_size,
                     metal_names, counter, directory, debugging, get_partners, trajectory_time, results)) for counter,
                     trajectory in enumerate(trajectories)]
        if len(trajectories) > 1:
            update_user('Analyzing {} frames from {} trajectories.'.format(total_number_of_frames, len(trajectories)),
                        logger)
        else:
            update_user('Analyzing {} frames from 1 trajectory.'.format(total_number_of_frames), logger)
        for chunk in chunks(processes, number_of_processes):
            for process in chunk:
                process.start()
            for process in chunk:
                process.join()
        update_user('Processing results.', logger)
        # convert multiprocessing list to true python list
        results_list = []
        for x in results:
            results_list.append(x)
        results = None
        dmif, partners = post_processing(results_list, total_number_of_frames)
        results_list = None
        update_user('Writing raw data to {}/data.'.format(directory), logger)
        pickle_writer(dmif, 'dmif', '{}/{}'.format(directory, 'data'))
        if get_partners:
            for key in grid_list_dict.keys():
                pickle_writer(partners[key].tolist(), key, '/'.join([directory, 'data']))
        partners = None
        update_user('Writing maps to {}/dmifs.'.format(directory), logger)
        for map_format in map_formats:
            for feature_type in [x for x in dmif.dtype.names if x not in ['x', 'y', 'z']]:
                dmif_writer(dmif[feature_type],
                            np.array([[x, y, z] for x, y, z in zip(dmif['x'], dmif['y'], dmif['z'])]),
                            map_format, feature_type, '{}/{}'.format(directory, 'dmifs'), logger)
    # generating exclusion volumes
    if config.has_section('exclusion volume parameters'):
        logger.debug('\n'.join([': '.join(list(_)) for _ in config.items('exclusion volume parameters')]))
        if 'dmif' not in locals():
            dmif = pickle_reader(config.get('exclusion volume parameters', 'dmif'), 'dmif', logger)
        shape_cutoff, restrictive = exclusion_volume_parameters(config)
        evs = generate_exclusion_volumes(dmif, directory, debugging, shape_cutoff, restrictive)
        pickle_writer(evs, 'exclusion_volumes', '/'.join([directory, 'data']))
    # generating features
    if config.has_section('feature parameters'):
        logger.debug('\n'.join([': '.join(list(_)) for _ in config.items('feature parameters')]))
        partner_path = None
        if 'dmif' not in locals():
            dmif = pickle_reader(config.get('feature parameters', 'dmif'), 'dmif', logger)
            partner_path = config.get('feature parameters', 'partners')
        features_per_feature_type, number_of_processes = feature_parameters(config)
        positions = np.array([[x, y, z] for x, y, z in zip(dmif['x'], dmif['y'], dmif['z'])])
        manager = multiprocessing.Manager()
        results = manager.list()
        feature_counter = multiprocessing.Value('i', 0)
        feature_time = time.time()
        processes = [multiprocessing.Process(target=generate_features, args=(positions, np.array(dmif[feature_type]),
                     feature_type, features_per_feature_type, directory, partner_path, debugging,
                     len(feature_types) * features_per_feature_type, feature_time, feature_counter, results)) for
                     feature_type in feature_types]
        update_user('Generating features.', logger)
        for chunk in chunks(processes, number_of_processes):
            for process in chunk:
                process.start()
            for process in chunk:
                process.join()
        update_user('Generated {} features.'.format(len(results)), logger)
        features = renumber_features(results)
        pickle_writer(features, 'features', '/'.join([directory, 'data']))
    # pharmacophore generation
    if config.has_section('pharmacophore parameters'):
        logger.debug('\n'.join([': '.join(list(_)) for _ in config.items('pharmacophore parameters')]))
        if config.has_option('pharmacophore parameters', 'features'):
            features = pickle_reader(config.get('pharmacophore parameters', 'features'), 'features', logger)
        if config.has_option('pharmacophore parameters', 'exclusion volumes'):
            evs = pickle_reader(config.get('pharmacophore parameters', 'exclusion volumes'), 'exclusion volumes',
                                logger)
        pharmacophore = renumber_features(features + evs)
        evs = [[counter + len(features) + 1] + x[1:] for counter, x in enumerate(evs)]
        pharmacophore_formats = pharmacophore_parameters(config)
        pharmacophore_directory = '/'.join([directory, 'pharmacophores'])
        update_user('Writing pharmacophore with all features to {}.'.format(pharmacophore_directory), logger)
        pharmacophore_writer(pharmacophore, pharmacophore_formats, 'super_pharmacophore', pharmacophore_directory,
                             logger)
    # library generation
    if config.has_section('library parameters'):
        logger.debug('\n'.join([': '.join(list(_)) for _ in config.items('library parameters')]))
        generate_library(*library_parameters(config, directory), directory, debugging)
    # dmif excess generation
    if config.has_section('dmif excess parameters'):
        dmif1_path, dmif2_path, dmif1_name, dmif2_name, map_formats = dmif_excess_parameters(config)
        dmif1_excess, dmif2_excess = generate_dmif_excess(dmif1_path, dmif2_path)
        update_user('Writing dmif excess maps to {0}/{1}_excess and {0}/{2}_excess.'.format(directory,
                    dmif1_name, dmif2_name), logger)
        for map_format in map_formats:
            for feature_type in [x for x in dmif1_excess.dtype.names if x not in ['x', 'y', 'z']]:
                dmif_writer(dmif1_excess[feature_type],
                            np.array([[x, y, z] for x, y, z in zip(dmif['x'], dmif['y'], dmif['z'])]),
                            map_format, feature_type,'{}/{}_excess'.format(directory, dmif1_name), logger)
                dmif_writer(dmif2_excess[feature_type],
                            np.array([[x, y, z] for x, y, z in zip(dmif['x'], dmif['y'], dmif['z'])]),
                            map_format, feature_type, '{}/{}_excess'.format(directory, dmif2_name), logger)
    # centroid generation
    if config.has_section('centroid parameters'):
        update_user('Starting screening of protein conformations.', logger)
        logger.debug('\n'.join([': '.join(list(_)) for _ in config.items('centroid parameters')]))
        ligand, pharmacophore_path, topology, trajectories, first_frame, last_frame, step_size, \
            total_number_of_frames, metal_names, output_name, number_of_processes = centroid_parameters(config,
                                                                                                        debugging)
        frame_counter = multiprocessing.Value('i', 0)
        trajectory_time = time.time()
        processes = [multiprocessing.Process(target=screen_protein_conformations, args=(topology, trajectory,
                     pharmacophore_path, ligand, counter, first_frame, last_frame, step_size, metal_names, directory,
                     output_name, debugging, total_number_of_frames, frame_counter, trajectory_time)) for counter,
                     trajectory in enumerate(trajectories)]
        if len(trajectories) > 1:
            update_user('Analyzing {} frames from {} trajectories.'.format(total_number_of_frames, len(trajectories)),
                        logger)
        else:
            update_user('Analyzing {} frames from 1 trajectory.'.format(total_number_of_frames), logger)
        for chunk in chunks(processes, number_of_processes):
            for process in chunk:
                process.start()
            for process in chunk:
                process.join()
        update_user('Finding centroid generation.', logger)
        if debugging:
            ensemble_to_centroid(topology, trajectories, output_name, directory, debugging)
        else:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                ensemble_to_centroid(topology, trajectories, output_name, directory, debugging)
        update_user('Output written to {0}.'.format('/'.join([directory, output_name])), logger)
    update_user('Finished after {}.'.format(time_to_text(time.time() - start_time)), logger)
