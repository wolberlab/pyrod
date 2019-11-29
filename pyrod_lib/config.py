""" PyRod - dynamic molecular interaction fields (dMIFs), based on tracing water molecules in MD simulations.

Released under the GNU Public Licence v2.

This module reads parameters from config files.
"""

# python standard libraries
import warnings

# external libraries
import MDAnalysis as mda

# pyrod modules
try:
    from pyrod.pyrod_lib.lookup import feature_types
    from pyrod.pyrod_lib.write import update_user
except ImportError:
    from pyrod_lib.lookup import feature_types
    from pyrod_lib.write import update_user


def test_grid_parameters(config):
    center = [int(x.strip()) for x in config.get('test grid parameters', 'center').split(',')]
    edge_lengths = [int(x.strip()) for x in config.get('test grid parameters', 'edge lengths').split(',')]
    name = '_'.join(str(_) for _ in center + edge_lengths)
    return [center, edge_lengths, name]


def point_properties_parameters(config):
    point = [float(x.strip()) for x in config.get('point properties parameters', 'point').split(',')]
    dmif_path = config.get('point properties parameters', 'dmif')
    return point, dmif_path


def trajectory_analysis_parameters(config, debugging):
    center = [int(x.strip()) for x in config.get('trajectory analysis parameters', 'center').split(',')]
    edge_lengths = [int(x.strip()) for x in config.get('trajectory analysis parameters', 'edge lengths').split(',')]
    topology = config.get('trajectory analysis parameters', 'topology')
    trajectories = [x.strip() for x in config.get('trajectory analysis parameters', 'trajectories').split(',')]
    first_frame = None
    if len(config.get('trajectory analysis parameters', 'first frame')) > 0:
        first_frame = int(config.get('trajectory analysis parameters', 'first frame')) - 1  # convert to zero-based
    last_frame = None
    if len(config.get('trajectory analysis parameters', 'last frame')) > 0:
        last_frame = int(config.get('trajectory analysis parameters', 'last frame'))
    step_size = None
    if len(config.get('trajectory analysis parameters', 'step size')) > 0:
        step_size = int(config.get('trajectory analysis parameters', 'step size'))
    if debugging:
        total_number_of_frames = len(mda.Universe(trajectories[0]).trajectory[first_frame:last_frame:step_size]) * \
            len(trajectories)
    else:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            total_number_of_frames = len(mda.Universe(trajectories[0]).trajectory[first_frame:last_frame:step_size]) * \
                len(trajectories)
    metal_names = [x.strip() for x in config.get('trajectory analysis parameters', 'metal names').split(',')]
    map_formats = []
    if len(config.get('trajectory analysis parameters', 'map formats')) > 0:
        map_formats = [x.strip() for x in config.get('trajectory analysis parameters', 'map formats').split(',')]
    number_of_processes = int(config.get('trajectory analysis parameters', 'number of processes'))
    get_partners = True
    if config.has_option('trajectory analysis parameters', 'dmifs only'):
        if config.get('trajectory analysis parameters', 'dmifs only') == 'true':
            get_partners = False
    return [center, edge_lengths, topology, trajectories, first_frame, last_frame, step_size, total_number_of_frames,
            metal_names, map_formats, number_of_processes, get_partners]


def exclusion_volume_parameters(config):
    shape_cutoff = float(config.get('exclusion volume parameters', 'shape cutoff'))
    restrictive = config.get('exclusion volume parameters', 'restrictive')
    if restrictive == 'true':
        restrictive = True
    else:
        restrictive = False
    return [shape_cutoff, restrictive]


def feature_parameters(config):
    features_per_feature_type = int(config.get('feature parameters', 'features per feature type'))
    number_of_processes = int(config.get('feature parameters', 'number of processes'))
    return [features_per_feature_type, number_of_processes]


def pharmacophore_parameters(config):
    pharmacophore_formats = [x.strip() for x in config.get('pharmacophore parameters',
                                                           'pharmacophore formats').split(',')]
    return pharmacophore_formats


def library_parameters(config, directory):
    pharmacophore_path = config.get('library parameters', 'pharmacophore path')
    output_format = config.get('library parameters', 'output format')
    library_dict = {}
    for parameter in ['minimal features', 'maximal features', 'minimal hydrogen bonds', 'maximal hydrogen bonds',
                      'minimal hydrophobic interactions', 'maximal hydrophobic interactions', 
                      'minimal aromatic interactions', 'maximal aromatic interactions', 
                      'minimal ionizable interactions', 'maximal ionizable interactions']:
        library_dict[parameter] = int(config.get('library parameters', parameter))
    if len(config.get('library parameters', 'library name')) > 0:
        library_name = config.get('library parameters', 'library name')
    else:
        library_name = 'library'
    library_path = '{}/pharmacophores/{}'.format(directory, library_name)
    pyrod_pharmacophore = True
    if config.get('library parameters', 'pyrod pharmacophore') == 'false':
        pyrod_pharmacophore = False
    return [pharmacophore_path, output_format, library_dict, library_path, pyrod_pharmacophore]


def dmif_excess_parameters(config):
    dmif1_path = config.get('dmif excess parameters', 'dmif 1')
    dmif2_path = config.get('dmif excess parameters', 'dmif 2')
    dmif1_name = config.get('dmif excess parameters', 'dmif 1 name')
    if len(dmif1_name) == 0:
        dmif1_name = 'dmif1'
    dmif2_name = config.get('dmif excess parameters', 'dmif 2 name')
    if len(dmif2_name) == 0:
        dmif2_name = 'dmif2'
    map_formats = []
    if len(config.get('dmif excess parameters', 'map formats')) > 0:
        map_formats = [x.strip() for x in config.get('dmif excess parameters', 'map formats').split(',')]
    return [dmif1_path, dmif2_path, dmif1_name, dmif2_name, map_formats]


def centroid_parameters(config, debugging):
    ligand = config.get('centroid parameters', 'ligand')
    pharmacophore = config.get('centroid parameters', 'pharmacophore')
    topology = config.get('centroid parameters', 'topology')
    trajectories = [x.strip() for x in config.get('centroid parameters', 'trajectories').split(',')]
    first_frame = None
    if len(config.get('centroid parameters', 'first frame')) > 0:
        first_frame = int(config.get('centroid parameters', 'first frame')) - 1  # convert to zero-based
    last_frame = None
    if len(config.get('centroid parameters', 'last frame')) > 0:
        last_frame = int(config.get('centroid parameters', 'last frame'))
    step_size = None
    if len(config.get('centroid parameters', 'step size')) > 0:
        step_size = int(config.get('centroid parameters', 'step size'))
    if debugging:
        total_number_of_frames = len(mda.Universe(trajectories[0]).trajectory[first_frame:last_frame:step_size]) * \
                                 len(trajectories)
    else:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            total_number_of_frames = len(mda.Universe(trajectories[0]).trajectory[first_frame:last_frame:step_size]) * \
                                     len(trajectories)
    metal_names = [x.strip() for x in config.get('centroid parameters', 'metal names').split(',')]
    if len(config.get('centroid parameters', 'output name')) > 0:
        output_name = config.get('centroid parameters', 'output name')
    else:
        output_name = 'centroid'
    number_of_processes = int(config.get('centroid parameters', 'number of processes'))
    return [ligand, pharmacophore, topology, trajectories, first_frame, last_frame, step_size, total_number_of_frames,
            metal_names, output_name, number_of_processes]
